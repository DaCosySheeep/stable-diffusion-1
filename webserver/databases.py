import psycopg2, psycopg2.extras

def checkque():
    try:
        conn=psycopg2.connect(host="192.168.86.10", database="sd", user="sd", password="password")
        cur=conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        #cur.row_factory=psycopg2.extras.DictCursor
        cur.execute(f"""
                    SELECT interactionid, prompt, negative_prompt, height, width, seed, safety_filter, n_samples, n_iter, ddim_steps, model_id, image_path
                    FROM public.queue
                    WHERE NOT public.queue.generated
                    AND NOT queue."raised error"
                    ORDER BY interactionid ASC
                    LIMIT 1;
                    """)
        result=cur.fetchone()
        #print(result)
        return result
    except Exception as e:
        print(e)
        return None
    finally:
        cur.close()
        conn.close()

def get_model_path_from_id(id):
    try:
        conn=psycopg2.connect(host="192.168.86.10", database="sd", user="sd", password="password")
        cur=conn.cursor()
        cur.execute(f"""
                    SELECT model_path
                    FROM public.models
                    WHERE model_id={id};
                    """)
        return cur.fetchone()
    except Exception as e:
        print(e)
        return None
    finally:
        cur.close()
        conn.close()

def set_completed(interactionid, generated=False, raised_error=False, safety_error=False, start_time="0", finish_time="0"):
    try:
        conn=psycopg2.connect(host="192.168.86.10", database="sd", user="sd", password="password")
        cur=conn.cursor()
        cur.execute(f"""
                    UPDATE public.queue
                    SET generated = {generated}, "raised error" = {raised_error}, "safety error" = {safety_error}, "time_started" = to_timestamp({start_time}), "time_finished" = to_timestamp({finish_time})
                    WHERE interactionid = {interactionid}

                    """)
        conn.commit()
        #return cur.fetchone()
    except Exception as e:
        print(e)
        return None
    finally:
        cur.close()
        conn.close()