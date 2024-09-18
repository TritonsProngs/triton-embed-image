import asyncio
import base64
import numpy as np
import json

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Triton Inference Server deployment utilizing the python_backend for image embedding
    models. Currently only SigLIP is supported.
    """

    def initialize(self, args):
        """
        Initialize embedding models' processors and load configuration parameters.

        Parameters
        ----------
        args : dict
            Command-line arguments for launching Triton Inference Server
        """
        # List of supported embedding models
        self.embed_models = set(["siglip_vision"])

        # Load the model configuration
        self.model_config = model_config = json.loads(args["model_config"])

        # Get the default embedding model. Can be overriden in request parameter
        self.default_embed_model = model_config["parameters"]["default_embed_model"][
            "string_value"
        ]
        
        if self.default_embed_model not in self.embed_models:
            raise pb_utils.TritonError(
                f"Invalid default embedding model: {self.default_embed_model}"
            )
                
        ## Get additional parameters from the config.pbtxt file
        # bool_value doesn't appear supported forcing using string_value
        # Specify the default value for base64_encoded request parameter.
        default_base64_encoded_str = model_config["parameters"][
            "default_base64_encoded"
        ]["string_value"]
        if default_base64_encoded_str.lower() == "true":
            self.default_base64_encoded = True
        elif default_base64_encoded_str.lower() == "false":
            self.default_base64_encoded = False
        else:
            raise pb_utils.TritonError(
                "model_config['parameters']['default_base64_encoded']="
                + f"{default_base64_encoded_str} must be 'true' | 'false'. "
            )


    def process_request(self, request, base64_encoded: bool):
        """
        Process the input image request and prepare pixel values for embedding.

        Parameters
        ----------
        request : pb_utils.InferenceRequest
            Inference request containing the input image.
        base64_encoded : bool
            Whether the input image is base64 encoded.

        Returns
        -------
        np.ndarray
            Processed pixel values of the input image.
        """
        try:
            input_image_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
        except Exception as exc:
            raise ValueError(f"Failed on getting input tensor from request: {exc}")

        if base64_encoded:
            try:
                images_bytes = []
                for b in input_image_tt.as_numpy().reshape(-1):
                    images_bytes.append(base64.b64decode(b))
                images_np = np.array(images_bytes, np.object_).reshape(-1, 1)
                input_image_tt = pb_utils.Tensor("INPUT_IMAGE", images_np)
            except Exception as exc:
                raise ValueError(
                    f"Failed on base64 decoding the image from request data: {exc}"
                )
        return input_image_tt

    async def execute(self, requests: list) -> list:
        """
        Execute a batch of embedding requests on provided images. Images can be
        sent either as raw bytes or base64 encoded strings.

        Option Request Parameters
        -------------------------
        embed_model : str
            Specify which embedding model to use.
            If None, default_embed_model is used.
        base64_encoded : bool
            Set to true if image is sent base64 encoded.
            If None, default_base64_encoded is used.


        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]
            List of inference requests each containing an image to be embedded.

        Returns
        -------
        List[pb_utils.InferenceResponse]
            List of response objects with embedding results or error messages
        """
        logger = pb_utils.Logger
        batch_size = len(requests)
        logger.log_info(f"embed_image.execute received {batch_size} requests")
        responses = [None] * batch_size
        inference_response_awaits = []
        valid_requests = []
        for batch_id, request in enumerate(requests):
            # Handle any request parameters
            request_params = json.loads(request.parameters())
            embed_model = request_params.get("embed_model", self.default_embed_model)
            base64_encoded = request_params.get(
                "base64_encoded", self.default_base64_encoded
            )

            if embed_model not in self.embed_models:
                responses[batch_id] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"Invalid embedding model: {embed_model}"
                    )
                )
                continue
            if base64_encoded not in [True, False]:
                responses[batch_id] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"Invalid base64_encoded: {base64_encoded}. "
                        + "Must be true or false."
                    )
                )
                continue

            try:
                input_image_tt = self.process_request(request, base64_encoded)
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"{exc}")
                )
                responses[batch_id] = response
                continue
            else:
                infer_model_request = pb_utils.InferenceRequest(
                    model_name=embed_model,
                    requested_output_names=["EMBEDDING"],
                    inputs=[input_image_tt],
                )
                # Perform asynchronous inference request
                inference_response_awaits.append(infer_model_request.async_exec())
                valid_requests.append(batch_id)

        inference_responses = await asyncio.gather(*inference_response_awaits)
        for model_response, batch_id in zip(inference_responses, valid_requests):
            if model_response.has_error() and responses[batch_id] is None:
                err_msg = (
                    "Error embedding the image: "
                    + f"{model_response.error().message()}"
                )
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(err_msg)
                )
                responses[batch_id] = response
            else:
                embedding_tt = pb_utils.get_output_tensor_by_name(
                    model_response, "EMBEDDING"
                )
                response = pb_utils.InferenceResponse(output_tensors=[embedding_tt])
                responses[batch_id] = response

        return responses
