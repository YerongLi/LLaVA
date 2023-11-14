import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    folder='/scratch/yerong/self-instruct/pipe/img'
    file_list = os.listdir(folder)
    prompt_dict = {
    'attention':"""You are an speech-language pathologist (SLP) and you are experienced in training kids with speech and language delay. Now you need to analyze a video frame where a SLP teaches a group of kids by applying the “Applied Behavioral Analysis (ABA)” method. You will detect if the children (circled) in this frame demonstrate any types of the four behaviors introduced below.
    Attention Seeking: Attention-seeking behavior occurs when someone desires feedback or a response from another person. Crying and throwing tantrums are great examples of childhood attention-seeking habits. Attention seekers may settle for any type of attention, whether positive or negative. Examples of attention include: Praise, such as cheering and words of affirmation; Scolding, saying no, or moving a child’s hand away; Redirecting your attention to your child; or Showing disappointment or frustration with facial expressions and body language.""",
    'escape':"""You are an speech-language pathologist (SLP) and you are experienced in training kids with speech and language delay. Now you need to analyze a video frame where a SLP teaches a group of kids by applying the “Applied Behavioral Analysis (ABA)” method. You will detect if the children (circled) in this frame demonstrate any types of the four behaviors introduced below.
    Escape or avoidance: Escape behaviors typically occur when a learner wants to avoid or “escape” doing something. This is common in ABA therapy session instructional periods.  For example, if a child does not want to complete a puzzle or read a book, she or he may run away from the therapist to avoid the instructional activity.""",
    'tangibles':"""You are an speech-language pathologist (SLP) and you are experienced in training kids with speech and language delay. Now you need to analyze a video frame where a SLP teaches a group of kids by applying the “Applied Behavioral Analysis (ABA)” method. You will detect if the children (circled) in this frame demonstrate any types of the four behaviors introduced below.
    Access to tangibles or reinforcements: Access to tangibles is somewhat self-explanatory, but it is also very important. Children may engage in certain behaviors because they are looking to gain access to something. For example, wanting a cookie. Keep in mind that access-related behaviors occur surrounding items the child can’t access independently. When trying to access a tangible reward, a child may: Beg, Scream, cry, or throw a tantrum, Hit or bite, Grab the item away from someone else.""",
    'sensory':"""You are an speech-language pathologist (SLP) and you are experienced in training kids with speech and language delay. Now you need to analyze a video frame where a SLP teaches a group of kids by applying the “Applied Behavioral Analysis (ABA)” method. You will detect if the children (circled) in this frame demonstrate any types of the four behaviors introduced below.
    Sensory needs: Sensory stimulation (also known as sensory needs) occurs when children want to experience a pleasant sensation or replace discomfort. Children may also seek stimulation to sensitize or desensitize, depending on their sensory needs. Sensory stimulation can manifest itself in several ways, such as: Jumping, Skipping, Hand-flapping, Tapping feet, Rocking back and forth.""",
    }
    # Iterate over each file in the folder
    for file_name in file_list:
        # Construct the full path to the image file
        image_path = os.path.join(folder, file_name)        
        image = load_image(image_path)
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        for key, inp in prompt_dict.items():
            if not inp:
                print("exit...")
                # break

            print(f"{roles[1]}: ", end="")

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            print(filename)
            with open(f'{filename.split(".")[0]}_{key}.txt', 'w') as file:
                file.write(prompt)
                file.write(outputs)
            conv.messages[-1][-1] = outputs

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
