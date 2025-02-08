import os
from transformers import AutoProcessor, VisionEncoderDecoderModel
import torch
from typing import Optional, List
import io
import fitz
from pathlib import Path
from PIL import Image

from transformers import StoppingCriteria, StoppingCriteriaList
from collections import defaultdict

# maximum length depends on the model, for the small model it is 3584
MAX_LENGTH = 3584

class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return
        if self.norm:
            return torch.var(self.values, 1) / self.values.shape[1]
        else:
            return torch.var(self.values, 1)


class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, MAX_LENGTH)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0
    










class PdfToText:
    def __init__(self, processor="facebook/nougat-small", 
                       model="facebook/nougat-small"):
        """
        :param processor: The name of the processor to use for encoding images.
            Defaults to "facebook/nougat-small".
        :param model: The name of the model to use for text generation.
            Defaults to "facebook/nougat-small".

        Initializes Nougat with the given processor and model.
        """
        self._processor_name = processor
        self._model_name = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """
        Load the Nougat model and processor.

        Loads the processor and model from HuggingFace's model hub based on the
        names specified in the constructor.

        :return: None
        """
        self._processor = AutoProcessor.from_pretrained(self._processor_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self._model_name)

        self._model.to(self.device)
    
    @property
    def model_name(self):
        """
        The name of the model used for text generation.

        :return: The model name
        :rtype: str
        """
        return self._model_name

    @staticmethod
    def rasterize_paper(pdf: Path,outpath: Optional[Path] = None,dpi: int = 96, return_pil=False, pages=None,) -> Optional[List[io.BytesIO]]:
        """
        Rasterize a PDF file to PNG images.

        Args:
            pdf (Path): The path to the PDF file.
            outpath (Optional[Path], optional): The output directory. If None, the PIL images will be returned instead. Defaults to None.
            dpi (int, optional): The output DPI. Defaults to 96 (from paper).
            return_pil (bool, optional): Whether to return the PIL images instead of writing them to disk. Defaults to False.
            pages (Optional[List[int]], optional): The pages to rasterize. If None, all pages will be rasterized. Defaults to None.

        Returns:
            Optional[List[io.BytesIO]]: The PIL images if `return_pil` is True, otherwise None.
        """

        pillow_images = []
        if outpath is None:
            return_pil = True
        try:
            if isinstance(pdf, (str, Path)):
                pdf = fitz.open(pdf)
            if pages is None:
                pages = range(len(pdf))
            for i in pages:
                page_bytes: bytes = pdf[i].get_pixmap(dpi=dpi).pil_tobytes(format="PNG")
                if return_pil:
                    pillow_images.append(io.BytesIO(page_bytes))
                else:
                    with (outpath / ("%02d.png" % (i + 1))).open("wb") as f:
                        f.write(page_bytes)
        except Exception:
            pass
        if return_pil:
            return pillow_images
    
    def pdf_to_image(self, file_path:Path, **kwargs):

        """
        Convert a PDF file to an image.

        Args:
            file_path (Path): The path to the PDF file.
    
            **kwargs: Additional keyword arguments to pass to rasterize_paper.

        Returns:
            List[Image.Image]: A list of images.
        """
        images = PdfToText.rasterize_paper(pdf=file_path, **kwargs)
        images = [Image.open(image) for image in images]
        return images 
       
    def convert(self, file_path:Path, pages=None):
        if (not hasattr(self, "_processor")) or (not hasattr(self, "_model")):
            self.load_model()

        images = self.pdf_to_image(file_path, pages=pages)
        pixel_values = self._processor(images=images, return_tensors="pt").pixel_values

        outputs = self._model.generate( pixel_values.to(self.device),
                                        min_length=1,
                                        max_length=MAX_LENGTH,
                                        bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                                        return_dict_in_generate=True,
                                        output_scores=True,
                                        stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]) )
        
        generated = self._processor.batch_decode(outputs[0], skip_special_tokens=True)
        generated = self._processor.post_process_generation(generated, fix_markdown=False)
        return generated
    


if __name__ == "__main__":
    # The path of the pdf file
    file_path = Path("C:/Users/abdel/Downloads/rapport1.pdf")

    # Initialize the PdfToText class
    pdfToText = PdfToText()

    # Convert the PDF to text
    generated_text = pdfToText.convert(file_path)

    # Create the output file path by replacing the extension
    output_path = file_path.with_suffix(".txt")

    # Write the generated content to the new text file
    try:
        with output_path.open("w", encoding="utf-8") as file:
            file.write("".join(generated_text))  # Concatenate all elements (pages) of `generated_text` into a single string
        print(f"Text successfully written to: {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")









