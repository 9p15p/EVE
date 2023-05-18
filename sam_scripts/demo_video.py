import numpy as np
import gradio as gr
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.util_sam.demo_utils import \
    Logger, \
    read_logs, \
    seg_everything, \
    init_EVE, \
    get_meta_from_video, \
    add_clicks, \
    seg_clicks, \
    reset_curr, \
    add_new_obj_click, \
    reset_video, \
    next_frame, \
    last_frame, \
    run_all

DEBUG_MODE = False if os.getenv("DEBUG_MODE") is None else True
print(f"DEBUG_MODE: {DEBUG_MODE}")

sys.stdout = Logger("output.log")

with gr.Blocks() as demo:
    #################create page#################
    gr.Markdown("# 🍎EVE")
    with gr.Accordion(label="Input Video", open=False):
        input_video = gr.Video(label="Input Video").style(height=480, width=910)
    with gr.Row():
        curr_overlay = gr.Image(label="Curr Frame", interactive=True).style(height=480, width=910)
        with gr.Column():
            with gr.Accordion(label="Logs", open=False):
                logs = gr.Textbox(label="Logs", max_lines=5000)
            if not DEBUG_MODE:
                demo.load(read_logs, None, logs, every=0.1)
            else:
                refresh_but = gr.Button(value="Refresh Logs.",
                                        interactive=True)
                refresh_but.click(
                    fn=read_logs,
                    inputs=None,
                    outputs=logs
                )

    with gr.Row():
        init_EVE_but = gr.Button(value="Init EVE",
                                 interactive=True)
        reset_curr_but = gr.Button(value="Reset Current frame.",
                                   interactive=True)
        reset_v_but = gr.Button(value="Reset Video",
                                interactive=True)
        last_frame_but = gr.Button(value="Last Frame",
                                   interactive=True)
        next_frame_but = gr.Button(value="Next Frame",
                                   interactive=True)
        run_all_but = gr.Button(value="Run All Frames and Save Them",
                                interactive=True)

    """click tab"""
    tab_click = gr.Tab(label="Click")
    with tab_click:
        with gr.Row():
            point_mode = gr.Radio(choices=["Single Box", "Positive Points", "Negative Points"],
                                  value="Single Box",
                                  label="Prompt",
                                  interactive=True)
            with gr.Column():
                seg_one_click_but = gr.Button(value="Seg one object.",
                                              interactive=True)

                add_new_obj_click_but = gr.Button(value="Add a new object",
                                                  interactive=True)

    """everything tab"""
    tab_everything = gr.Tab(label="Everything")
    with tab_everything:
        with gr.Row():
            seg_every_curr_but = gr.Button(value="Seg everything on the current frame.",
                                           interactive=True)
            with gr.Column():
                MAX_NUM_OBJ = gr.Slider(
                    label='MAX_NUM_OBJ',
                    minimum=1,
                    step=1,
                    maximum=100,
                    value=20,
                    interactive=True,
                )
                ALLOW_MIN_AREA = gr.Slider(
                    label='ALLOW_MIN_AREA',
                    minimum=0,
                    step=2000,
                    maximum=40000,
                    value=500,
                    interactive=True,
                )
    with gr.Row():
        with gr.Accordion(label="Outputs", open=False):
            with gr.Tab(label="Output Video"):
                output_video = gr.Video(label="Output Video").style(height=480, width=910)
            with gr.Tab(label="Predicted Overlays"):
                output_overlay = gr.File(label="Predicted Overlays")
            with gr.Tab(label="Predicted Masks"):
                output_mask = gr.File(label="Predicted Masks")

    #################opearation#################
    EVE, EVE_config, mapper, processor, auto_masker, prompt_predictor = [gr.State(None) for i in range(6)]
    curr_img = gr.State(None)
    masks_tensor = gr.State(None)  # 即当前帧mask,保留此名兼容之前的代码
    clicks_stack = gr.State(dict(box=[], point=[], point_lb=[]))
    imgs_stack = gr.State(list)
    msks_stack = gr.State(list)
    f_id = gr.State(0)

    input_video.change(
        fn=get_meta_from_video,
        inputs=[input_video],
        outputs=[EVE, EVE_config, processor, auto_masker, prompt_predictor,
                 mapper, f_id, imgs_stack, msks_stack, curr_img, masks_tensor, clicks_stack, curr_overlay]
    )
    init_EVE_but.click(
        fn=init_EVE,
        inputs=None,
        outputs=[EVE, EVE_config, processor, auto_masker, prompt_predictor]
    )
    reset_curr_but.click(
        fn=reset_curr,
        inputs=[curr_img],
        outputs=[masks_tensor, clicks_stack, curr_overlay],
    )
    reset_v_but.click(
        fn=reset_video,
        inputs=[input_video],
        outputs=[mapper, f_id, imgs_stack, msks_stack, curr_img, masks_tensor, clicks_stack, curr_overlay],
    )
    last_frame_but.click(
        fn=last_frame,
        inputs=[processor,
                f_id, imgs_stack, msks_stack],
        outputs=[processor,
                 f_id, msks_stack, curr_img, masks_tensor, clicks_stack, curr_overlay]
    )
    next_frame_but.click(
        fn=next_frame,
        inputs=[mapper, processor,
                f_id, imgs_stack, msks_stack, masks_tensor],
        outputs=[processor,
                 f_id, msks_stack, curr_img, masks_tensor, clicks_stack, curr_overlay]
    )
    run_all_but.click(
        fn=run_all,
        inputs=[mapper, processor,
                f_id, imgs_stack, msks_stack, masks_tensor],
        outputs=[processor,
                 f_id, msks_stack, curr_img, masks_tensor, clicks_stack, curr_overlay,
                 output_video, output_overlay, output_mask]
    )

    """everything tab"""
    seg_every_curr_but.click(
        fn=seg_everything,
        inputs=[curr_img, auto_masker, ALLOW_MIN_AREA, MAX_NUM_OBJ],
        outputs=[masks_tensor, curr_overlay],
    )

    """click tab"""
    curr_overlay.select(
        fn=add_clicks,
        inputs=[prompt_predictor, curr_overlay, clicks_stack, point_mode, ],
        outputs=[curr_overlay, clicks_stack]
    )
    seg_one_click_but.click(
        fn=seg_clicks,
        inputs=[prompt_predictor, curr_img, masks_tensor, clicks_stack],
        outputs=[masks_tensor, clicks_stack, curr_overlay],
    )

    add_new_obj_click_but.click(
        fn=add_new_obj_click,
        inputs=[prompt_predictor, curr_img, masks_tensor, clicks_stack],
        outputs=[masks_tensor, clicks_stack, curr_overlay],
    )

    # TODO:
    #  0.新增中间修改功能
    #  0.新增一个列表用于选择不同颜色的目标
    #  1.增加一个Radio，用于选择是哪个目标
    #  2.增加修改前后帧的功能
    #  3.确认Y19如何在中间帧增加新的目标

if __name__ == '__main__':
    if DEBUG_MODE:
        demo.launch(server_name='0.0.0.0', debug=True, enable_queue=True)
    else:
        demo.queue().launch(server_name='0.0.0.0', debug=True, enable_queue=True)

    # import gradio as gr
    #
    # with gr.Blocks() as demo:
    #     gr.Markdown("# Greetings from Gradio!")
    #     inp = gr.Textbox(placeholder="What is your name???")
    #     out = gr.Textbox()
    #
    #     inp.change(fn=lambda x: f"Welcome, {x}!",
    #                inputs=inp,
    #                outputs=out)
    #
    # if __name__ == "__main__":
    #     demo.launch()
