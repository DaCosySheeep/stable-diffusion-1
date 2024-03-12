#Stable diffusion webserver
from tkinter.messagebox import showerror, showinfo, showwarning
import sys, time, functools, os, datetime, builtins, threading, pystray, random, subprocess, torch, numpy, json, default_prompt, argparse

from scripts.txt2img_bot import main as generate
from databases import checkque, get_model_path_from_id, set_completed
global debug
debug=False
def main():
    os.system("title "+"SD generator")

    while True:
        result=checkque()
        if debug: print(result)
        if result:
            #generate the image
            try:
                start_time=int(datetime.datetime.now().timestamp())
                if debug: print(f"Start time: {start_time}")
                
                generate(prompt=result[1], negative_prompt=result[2], 
                        skip_grid=True, H=result[3], W=result[4],
                        seed=result[5], skip_safety_check=not result[6],
                        n_samples=result[7], n_iter=result[8],
                        ddim_steps=result[9], outdir='E:'+result[11],
                        ckpt=get_model_path_from_id(result[10])[0]
                        )
                finish_time=int(datetime.datetime.now().timestamp())
                if debug: print(f"Finish time: {finish_time}")
                set_completed(generated=True, raised_error=False, safety_error=False, start_time=start_time, finish_time=finish_time, interactionid=result[0])
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                set_completed(generated=False, raised_error=True, interactionid=result[0])
                print(e)
                showerror(title="SD generator", message=f"{e}")
        else:
            #wait a second and check again.
            time.sleep(1)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--debug",
                        action="store_true",
                        help="Should the console print extra debug information")
    opt=parser.parse_args()
    debug=opt.debug
    print(f"Debug: {debug}")
    main()