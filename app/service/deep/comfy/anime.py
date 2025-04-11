import os
import random
import sys
import json
import argparse
import contextlib
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        if args is None or args.comfyui_directory is None:
            path = os.getcwd()
        else:
            path = args.comfyui_directory

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        import __main__

        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")

        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes(init_custom_nodes=True)


def save_image_wrapper(context, cls):
    if args.output is None:
        return cls

    from PIL import Image, ImageOps, ImageSequence
    from PIL.PngImagePlugin import PngInfo

    import numpy as np

    class WrappedSaveImage(cls):
        counter = 0

        def save_images(
            self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
        ):
            if args.output is None:
                return super().save_images(
                    images, filename_prefix, prompt, extra_pnginfo
                )
            else:
                if len(images) > 1 and args.output == "-":
                    raise ValueError("Cannot save multiple images to stdout")
                filename_prefix += self.prefix_append

                results = list()
                for batch_number, image in enumerate(images):
                    i = 255.0 * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                    metadata = None
                    if not args.disable_metadata:
                        metadata = PngInfo()
                        if prompt is not None:
                            metadata.add_text("prompt", json.dumps(prompt))
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                    if args.output == "-":
                        # Hack to briefly restore stdout
                        if context is not None:
                            context.__exit__(None, None, None)
                        try:
                            img.save(
                                sys.stdout.buffer,
                                format="png",
                                pnginfo=metadata,
                                compress_level=self.compress_level,
                            )
                        finally:
                            if context is not None:
                                context.__enter__()
                    else:
                        subfolder = ""
                        if len(images) == 1:
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = "output.png"
                            else:
                                subfolder, file = os.path.split(args.output)
                                if subfolder == "":
                                    subfolder = os.getcwd()
                        else:
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = filename_prefix
                            else:
                                subfolder, file = os.path.split(args.output)

                            if subfolder == "":
                                subfolder = os.getcwd()

                            files = os.listdir(subfolder)
                            file_pattern = file
                            while True:
                                filename_with_batch_num = file_pattern.replace(
                                    "%batch_num%", str(batch_number)
                                )
                                file = (
                                    f"{filename_with_batch_num}_{self.counter:05}.png"
                                )
                                self.counter += 1

                                if file not in files:
                                    break

                        img.save(
                            os.path.join(subfolder, file),
                            pnginfo=metadata,
                            compress_level=self.compress_level,
                        )
                        print("Saved image to", os.path.join(subfolder, file))
                        results.append(
                            {
                                "filename": file,
                                "subfolder": subfolder,
                                "type": self.type,
                            }
                        )

                return {"ui": {"images": results}}

    return WrappedSaveImage


def parse_arg(s: Any):
    """Parses a JSON string, returning it unchanged if the parsing fails."""
    if __name__ == "__main__" or not isinstance(s, str):
        return s

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


parser = argparse.ArgumentParser(
    description="A converted ComfyUI workflow. Required inputs listed below. Values passed should be valid JSON (assumes string if not valid JSON)."
)
parser.add_argument(
    "--queue-size",
    "-q",
    type=int,
    default=1,
    help="How many times the workflow will be executed (default: 1)",
)

parser.add_argument(
    "--comfyui-directory",
    "-c",
    default=None,
    help="Where to look for ComfyUI (default: current directory)",
)

parser.add_argument(
    "--input",
    "-i",
    default=None,
    help="The location of the input image. should be a file path",
)

parser.add_argument(
    "--output",
    "-o",
    default=None,
    help="The location to save the output image. should be a file path",
)

parser.add_argument(
    "--max-size",
    "-s",
    type=int,
    default=1024,
    help="Max resolution of width and height",
)

parser.add_argument(
    "--device",
    "-d",
    default='CPU',
    help="Device, should be CUDA, CoreML, ROCM",
)

parser.add_argument(
    "--disable-metadata",
    action="store_true",
    help="Disables writing workflow metadata to the outputs",
)


comfy_args = [sys.argv[0]]
if __name__ == "__main__" and "--" in sys.argv:
    idx = sys.argv.index("--")
    comfy_args += sys.argv[idx + 1 :]
    sys.argv = sys.argv[:idx]

args = None
if __name__ == "__main__":
    args = parser.parse_args()
    sys.argv = comfy_args
if args is not None and args.output is not None and args.output == "-":
    ctx = contextlib.redirect_stdout(sys.stderr)
else:
    ctx = contextlib.nullcontext()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes(init_custom_nodes=True)


_custom_nodes_imported = False
_custom_path_added = False


def main(*func_args, **func_kwargs):
    global args, _custom_nodes_imported, _custom_path_added
    if __name__ == "__main__":
        if args is None:
            args = parser.parse_args()
    else:
        defaults = dict(
            (arg, parser.get_default(arg))
            for arg in ["queue_size", "comfyui_directory", "output", "disable_metadata"]
        )
        ordered_args = dict(zip([], func_args))

        all_args = dict()
        all_args.update(defaults)
        all_args.update(ordered_args)
        all_args.update(func_kwargs)

        args = argparse.Namespace(**all_args)

    if args.input is None or os.path.isfile(args.input) == False:
        print(f'input file: {args.input} not exist')
        sys.exit(1000) 
        
    if args.output is None or os.path.isdir(os.path.dirname(args.output)) == False:
        print(f'output dir: {args.output} not exist')
        sys.exit(1001) 
    
    max_size = args.max_size
    if max_size <= 512 :
        max_size = 512
    elif max_size >= 4096:
        max_size = 4096
        
    device = args.device
    if device not in ["CUDA", "CoreML", "ROCM"]:
        print(f'deivce ({device}) unsupport, fallback to CPU')
        device = 'CPU'
        
        
    with ctx:
        if not _custom_path_added:
            add_comfyui_directory_to_sys_path()
            add_extra_model_paths()

            _custom_path_added = True

        if not _custom_nodes_imported:
            import_custom_nodes()

            _custom_nodes_imported = True

        from nodes import NODE_CLASS_MAPPINGS

    with torch.inference_mode(), ctx:
        imageloader = NODE_CLASS_MAPPINGS["ImageLoader"]()
        imageloader_1 = imageloader.load_image(
            path=args.input #"/Users/wadahana/Desktop/output2.jpg"
        )

        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_3 = checkpointloadersimple.load_checkpoint(
            ckpt_name="MoyouV2.safetensors"
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_9 = cliptextencode.encode(
            text="disproportional, Octane render, smudge, blurred, Low resolution, worst quality",
            clip=get_value_at_index(checkpointloadersimple_3, 1),
        )

        constrainimagenode = NODE_CLASS_MAPPINGS["ConstrainImageNode"]()
        constrainimagenode_2 = constrainimagenode.constrain_image(
            max_width=max_size,
            max_height=max_size,
            min_width=0,
            min_height=0,
            crop_if_required="no",
            images=get_value_at_index(imageloader_1, 0),
        )

        wd14tagger = NODE_CLASS_MAPPINGS["WD14Tagger"]()
        wd14tagger_63 = wd14tagger.process(
            model="wd-v1-4-swinv2-tagger-v2",
            device=device,
            threshold=0.35,
            character_threshold=0.85,
            replace_underscore=False,
            trailing_comma=False,
            exclude_tags="",
            images=get_value_at_index(constrainimagenode_2, 0),
        )
        text = get_value_at_index(wd14tagger_63, 0)
        cliptextencode_10 = cliptextencode.encode(
            text=get_value_at_index(wd14tagger_63, 0),
            clip=get_value_at_index(checkpointloadersimple_3, 1),
        )

        ipadapterinsightfaceloader = NODE_CLASS_MAPPINGS["IPAdapterInsightFaceLoader"]()
        ipadapterinsightfaceloader_17 = ipadapterinsightfaceloader.load_insightface(
            provider=device
        )

        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        vaeencode_30 = vaeencode.encode(
            pixels=get_value_at_index(constrainimagenode_2, 0),
            vae=get_value_at_index(checkpointloadersimple_3, 2),
        )

        ultralyticsdetectorprovider = NODE_CLASS_MAPPINGS[
            "UltralyticsDetectorProvider"
        ]()
        ultralyticsdetectorprovider_45 = ultralyticsdetectorprovider.doit(
            model_name="bbox/face_yolov8m.pt"
        )

        ipadapterunifiedloader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()
        ipadapter = NODE_CLASS_MAPPINGS["IPAdapter"]()
        ipadapterunifiedloaderfaceid = NODE_CLASS_MAPPINGS[
            "IPAdapterUnifiedLoaderFaceID"
        ]()
        ipadapterfaceid = NODE_CLASS_MAPPINGS["IPAdapterFaceID"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        facedetailer = NODE_CLASS_MAPPINGS["FaceDetailer"]()
        imagesaver = NODE_CLASS_MAPPINGS["ImageSaver"]()
        for q in range(args.queue_size):
            ipadapterunifiedloader_13 = ipadapterunifiedloader.load_models(
                preset="PLUS (high strength)",
                model=get_value_at_index(checkpointloadersimple_3, 0),
            )

            ipadapter_14 = ipadapter.apply_ipadapter(
                weight=1,
                start_at=0,
                end_at=1,
                weight_type="style transfer",
                model=get_value_at_index(ipadapterunifiedloader_13, 0),
                ipadapter=get_value_at_index(ipadapterunifiedloader_13, 1),
                image=get_value_at_index(constrainimagenode_2, 0),
            )

            ipadapterunifiedloaderfaceid_16 = ipadapterunifiedloaderfaceid.load_models(
                preset="FACEID PLUS - SD1.5 only",
                lora_strength=0.6,
                provider=device,
                model=get_value_at_index(ipadapter_14, 0),
            )

            ipadapterfaceid_18 = ipadapterfaceid.apply_ipadapter(
                weight=1,
                weight_faceidv2=1,
                weight_type="linear",
                combine_embeds="concat",
                start_at=0,
                end_at=1,
                embeds_scaling="V only",
                model=get_value_at_index(ipadapterunifiedloaderfaceid_16, 0),
                ipadapter=get_value_at_index(ipadapterunifiedloaderfaceid_16, 1),
                image=get_value_at_index(constrainimagenode_2, 0),
                insightface=get_value_at_index(ipadapterinsightfaceloader_17, 0),
            )

            ksampler_31 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.4,
                model=get_value_at_index(ipadapter_14, 0),
                positive=get_value_at_index(cliptextencode_10, 0),
                negative=get_value_at_index(cliptextencode_9, 0),
                latent_image=get_value_at_index(vaeencode_30, 0),
            )

            vaedecode_29 = vaedecode.decode(
                samples=get_value_at_index(ksampler_31, 0),
                vae=get_value_at_index(checkpointloadersimple_3, 2),
            )

            facedetailer_46 = facedetailer.doit(
                guide_size=512,
                guide_size_for=True,
                max_size=1024,
                seed=random.randint(1, 2**64),
                steps=16,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.5,
                feather=8,
                noise_mask=True,
                force_inpaint=True,
                bbox_threshold=0.5,
                bbox_dilation=10,
                bbox_crop_factor=3,
                sam_detection_hint="center-1",
                sam_dilation=0,
                sam_threshold=0.93,
                sam_bbox_expansion=0,
                sam_mask_hint_threshold=0.7,
                sam_mask_hint_use_negative="False",
                drop_size=10,
                wildcard="detail face, beautiful face, eyes, ",
                cycle=1,
                inpaint_model=True,
                noise_mask_feather=20,
                image=get_value_at_index(vaedecode_29, 0),
                model=get_value_at_index(ipadapterfaceid_18, 0),
                clip=get_value_at_index(checkpointloadersimple_3, 1),
                vae=get_value_at_index(checkpointloadersimple_3, 2),
                positive=get_value_at_index(cliptextencode_10, 0),
                negative=get_value_at_index(cliptextencode_9, 0),
                bbox_detector=get_value_at_index(ultralyticsdetectorprovider_45, 0),
            )

            imagesaver_64 = imagesaver.save_image(
                path=args.output, #"/Users/wadahana/Desktop/output.jpg",
                quality=85,
                images=get_value_at_index(facedetailer_46, 0),
            )


if __name__ == "__main__":
    main()
