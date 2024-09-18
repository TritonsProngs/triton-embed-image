from io import BytesIO
import numpy as np
from PIL import Image
from transformers import SiglipProcessor

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Triton Inference Server deployment utilizing the python_backend for
    SiGLIP Vision Processor.
    """

    def initialize(self, args):
        """
        Initialize SiGLIP Vision Processor.

        Parameters
        ----------
        args : dict
            Command-line arguments for launching Triton Inference Server
        """
        self.processor = SiglipProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384", local_files=True
        )

    def process_request(self, request):
        """
        Process the input image request and prepare pixel values for embedding.

        Parameters
        ----------
        request : pb_utils.InferenceRequest
            Inference request containing the input image.

        Returns
        -------
        np.ndarray
            Processed pixel values of the input image.
        """
        try:
            input_image_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
        except Exception as exc:
            raise ValueError(f"Failed on getting input tensor from request: {exc}")

        try:
            images_bytes = []
            for b in input_image_tt.as_numpy().reshape(-1):
                images_bytes.append(b)
        except Exception as exc:
            raise ValueError(
                f"Failed getting bytes of the image from request data: {exc}"
            )

        try:
            images = [Image.open(BytesIO(b)).convert("RGB") for b in images_bytes]
        except Exception as exc:
            raise ValueError(f"Failed on PIL.Image.open() request data: {exc}")

        try:
            pixel_values_np = self.processor(
                images=images, padding="max_length", return_tensors="pt"
            )["pixel_values"].numpy()
        except Exception as exc:
            raise ValueError(
                f"Failed on SiglipProcessor(images=image): {exc}"
            )

        # Shape = [batch_size, 3, 384, 384], where batch_size should be 1
        return pixel_values_np


    def execute(self, requests: list) -> list:
        """
        Execute a batch of processing requests on provided images.

        Output Shape after processing = (3, 384, 384), dtype=np.float32

        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]
            List of inference requests, each containing an image to be processed.

        Returns
        -------
        List[pb_utils.InferenceResponse]
            List of response objects w/ pixel values of input image or error response.
        """
        logger = pb_utils.Logger
        batch_size = len(requests)
        logger.log_info(
            f"siglip_vision_process.execute received {batch_size} requests"
        )
        responses = [None] * batch_size
        for batch_id, request in enumerate(requests):
            try:
                pixel_values_np = self.process_request(request)
                pixel_values_tt = pb_utils.Tensor("PIXEL_VALUES", pixel_values_np)
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "PIXEL_VALUES", np.zeros((1, 3, 384, 384), dtype=np.float32)
                        ),
                    ],
                    error=pb_utils.TritonError(f"{exc}"),
                )
                responses[batch_id] = response
            else:
                response = pb_utils.InferenceResponse(
                    output_tensors=[pixel_values_tt]
                )
                responses[batch_id] = response

        return responses
