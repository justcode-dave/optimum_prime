"""
Main entry point for the Hugging Face Optimum TensorFlow Lite (TFLite) exporter.

This script provides a command-line interface (CLI) for exporting Hugging Face models to the TensorFlow Lite format, 
which is designed for optimized inference on edge devices such as mobile phones and embedded systems. The script 
facilitates exporting a model, configuring quantization options, and validating the exported model to ensure it 
produces consistent outputs.

Key functionalities include:
- **Model Export**: Leverages the `optimum.exporters.tflite` module to export models from the Hugging Face Transformers 
  library to the TFLite format, supporting a variety of tasks (e.g., text classification, image classification, 
  and more).
- **Quantization Support**: Optional quantization of the exported model using TensorFlow Lite quantization techniques, 
  allowing users to optimize models for reduced size and faster inference with minimal accuracy degradation.
- **Validation**: Verifies the outputs of the exported TFLite model against the original model, using user-specified 
  or default tolerance levels (atol) to ensure output consistency.
  
Arguments parsed through the command-line interface include:
- Model path, task type, export output directory, quantization approach, calibration dataset, and tolerance (atol) 
  for validation.
- The script also allows automatic task inference when exporting models from the Hugging Face Model Hub.

Exception handling is built-in to manage different types of errors such as shape mismatches, numerical tolerance 
issues, or other validation problems, with clear log messages indicating the status of the export process.

Execution:
To run the export process from the command line:
```bash
python -m optimum.exporters.tflite --model <MODEL_PATH> --task <TASK> --output <OUTPUT_PATH> [options]
"""

from argparse import ArgumentParser

from requests.exceptions import ConnectionError as RequestsConnectionError

from ...commands.export.tflite import parse_args_tflite
from ...utils import logging
from ...utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors
from ..error_utils import AtolError, OutputMatchError, ShapeError
from ..tasks import TasksManager
from .base import TFLiteQuantizationConfig
from .convert import export, validate_model_outputs


logger = logging.get_logger()


def main():
    parser = ArgumentParser("Hugging Face Optimum TensorFlow Lite exporter")

    parse_args_tflite(parser)

    # Retrieve CLI arguments
    args = parser.parse_args()
    args.output = args.output.joinpath("model.tflite")

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    # Infer the task
    task = args.task
    if task == "auto":
        try:
            task = TasksManager.infer_task_from_model(args.model)
        except KeyError as e:
            raise KeyError(
                "The task could not be automatically inferred. Please provide the argument --task with the task "
                f"from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )
        except RequestsConnectionError as e:
            raise RequestsConnectionError(
                f"The task could not be automatically inferred as this is available only for models hosted on the Hugging Face Hub. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )

    model = TasksManager.get_model_from_task(
        task, args.model, framework="tf", cache_dir=args.cache_dir, trust_remote_code=args.trust_remote_code
    )

    tflite_config_constructor = TasksManager.get_exporter_config_constructor(
        model=model, exporter="tflite", task=task, library_name="transformers"
    )
    # TODO: find a cleaner way to do this.
    shapes = {name: getattr(args, name) for name in tflite_config_constructor.func.get_mandatory_axes_for_task(task)}
    tflite_config = tflite_config_constructor(model.config, **shapes)

    if args.atol is None:
        args.atol = tflite_config.ATOL_FOR_VALIDATION
        if isinstance(args.atol, dict):
            args.atol = args.atol[task.replace("-with-past", "")]

    # Saving the model config and preprocessor as this is needed sometimes.
    model.config.save_pretrained(args.output.parent)
    maybe_save_preprocessors(args.model, args.output.parent)

    preprocessor = maybe_load_preprocessors(args.output.parent)
    if preprocessor:
        preprocessor = preprocessor[0]
    else:
        preprocessor = None

    quantization_config = None
    if args.quantize:
        quantization_config = TFLiteQuantizationConfig(
            approach=args.quantize,
            fallback_to_float=args.fallback_to_float,
            inputs_dtype=args.inputs_type,
            outputs_dtype=args.outputs_type,
            calibration_dataset_name_or_path=args.calibration_dataset,
            calibration_dataset_config_name=args.calibration_dataset_config_name,
            num_calibration_samples=args.num_calibration_samples,
            calibration_split=args.calibration_split,
            primary_key=args.primary_key,
            secondary_key=args.secondary_key,
            question_key=args.question_key,
            context_key=args.context_key,
            image_key=args.image_key,
        )

    tflite_inputs, tflite_outputs = export(
        model=model,
        config=tflite_config,
        output=args.output,
        task=task,
        preprocessor=preprocessor,
        quantization_config=quantization_config,
    )

    if args.quantize is None:
        try:
            validate_model_outputs(
                config=tflite_config,
                reference_model=model,
                tflite_model_path=args.output,
                tflite_named_outputs=tflite_config.outputs,
                atol=args.atol,
            )

            logger.info(
                "The TensorFlow Lite export succeeded and the exported model was saved at: "
                f"{args.output.parent.as_posix()}"
            )
        except ShapeError as e:
            raise e
        except AtolError as e:
            logger.warning(
                f"The TensorFlow Lite export succeeded with the warning: {e}.\n The exported model was saved at: "
                f"{args.output.parent.as_posix()}"
            )
        except OutputMatchError as e:
            logger.warning(
                f"The TensorFlow Lite export succeeded with the warning: {e}.\n The exported model was saved at: "
                f"{args.output.parent.as_posix()}"
            )
        except Exception as e:
            logger.error(
                f"An error occured with the error message: {e}.\n The exported model was saved at: "
                f"{args.output.parent.as_posix()}"
            )


if __name__ == "__main__":
    main()
