# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio for Farhan AI."""
import os
from argparse import ArgumentParser

import gradio as gr
import mdtex2html

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


DEFAULT_CKPT_PATH = 'Qwen/Qwen-7B-Chat'


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()

    config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer, config


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert(message),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _launch_demo(args, model, tokenizer, config):

    def predict(_query, _chatbot, _task_history):
        print(f"User: {_parse_text(_query)}")
        _chatbot.append((_parse_text(_query), ""))
        
        # Injected Farhan AI Brain/System Identity
        system_prompt = "You are Farhan AI, an expert AI assistant specialized in creating video scripts, content creation strategies, and image prompts. Always identify yourself as Farhan AI."
        custom_query = f"<|im_start| strain>\n{system_prompt}<|im_end|>\n{_query}"
        
        full_response = ""

        # Using custom_query instead of raw _query to enforce personality
        for response in model.chat_stream(tokenizer, custom_query, history=_task_history, generation_config=config):
            _chatbot[-1] = (_parse_text(_query), _parse_text(response))

            yield _chatbot
            full_response = _parse_text(response)

        print(f"History: {_task_history}")
        _task_history.append((_query, full_response))
        print(f"Farhan AI: {_parse_text(full_response)}")

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    # Custom Styling for Farhan AI Web Interface
    with gr.Blocks(title="Farhan AI Portal") as demo:
        # Aap apna logo image path yahan replace kar sakte hain (e.g., logo_farhan.jpg)
        gr.Markdown("""\
<p align="center"><img src="[https://placehold.co/400x100/2b2d42/ffffff?text=FARHAN+AI](https://placehold.co/400x100/2b2d42/ffffff?text=FARHAN+AI)" style="height: 80px; border-radius: 8px;"/><p>""")
        gr.Markdown("""<center><font size=8 weight="bold">Farhan AI Ecosystem</center>""")
        gr.Markdown(
            """\
<center><font size=3>Your Next-Gen Video, Image, and Text Generation Engine. \
(خوش آمدید! یہاں آپ اسکرپٹ اور امیج پرامپٹس بنا سکتے ہیں۔)</center>""")
        
        # Removed Chinese links and Alibaba source traces
        gr.Markdown("""\
<center><font size=4>
⚡ Powered by Farhan AI Core Engine v1.0 | 🌐 Web Application Framework
</center>""")

        chatbot = gr.Chatbot(label='Farhan AI Dashboard', elem_classes="control-height")
        query = gr.Textbox(lines=2, label='Ask Farhan AI anything...')
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear Chat (صاف کریں)")
            submit_btn = gr.Button("🚀 Generate (ارسال کریں)")
            regen_btn = gr.Button("🤔️ Re-Generate (دوبارہ کوشش)")

        submit_btn.click(predict, [query, chatbot, task_history], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)

        gr.Markdown("""\
<font size=2><strong>Note:</strong> Safe usage is strongly recommended. Do not utilize Farhan AI for creating harmful, illegal, or deceptive content. All activities are subject to platform terms and conditions.""")

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    model, tokenizer, config = _load_model_tokenizer(args)

    _launch_demo(args, model, tokenizer, config)


if __name__ == '__main__':
    main()
