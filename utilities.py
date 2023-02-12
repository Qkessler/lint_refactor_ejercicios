import pandas as pd
import io
import json
import os
 
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders

from azure.storage.blob import ContainerClient

from forma_LA.constants import (
    VISITS_FOLDER,
    BLOB_CONTAINER_NAME,
    BLOB_CONTAINER_PROCESSED_NAME,
    INTERMEDIATE_CONTAINER_NAME,
    BACKUP_INTERMEDIATE_CONTAINER_NAME,
)
from forma_LA.container import Container

import re


def select_user(df, user, drop_user_field=False):
    selected_df = df[df["user"] == user].copy()

    if drop_user_field:
        if isinstance(df.columns, pd.MultiIndex):
            selected_df.drop(columns="user", inplace=True, level=0)
        else:
            selected_df.drop(columns="user", inplace=True)

    return selected_df


def group_df_dict(df, grouping_vars, dict_vars, column_name):
    grouped_df = df.groupby(grouping_vars, group_keys=True)

    df_dict_column = pd.DataFrame(
        (grouped_df.apply(lambda x: x[dict_vars].to_dict(orient="records"))),
        columns=[column_name],
    ).reset_index()

    return df_dict_column


def curate_video_df(video_viz):
    video_viz_df = video_viz.copy()
    video_viz_df = video_viz_df.astype(str)

    df_col_dict_3 = group_df_dict(
        video_viz_df,
        [
            "activity",
            "current_title",
            "element",
            "description",
            "duration",
            "user",
            "email",
            "name",
        ],
        dict_vars=["position", "viz", "formatted_position"],
        column_name="video_viz",
    )

    df_col_dict_2 = group_df_dict(
        df_col_dict_3,
        ["activity", "current_title", "element", "description", "duration"],
        dict_vars=["user", "email", "name", "video_viz"],
        column_name="users",
    )

    df_col_dict_1 = group_df_dict(
        df_col_dict_2,
        ["activity", "current_title"],
        ["element", "description", "duration", "users"],
        column_name="videos",
    )

    return df_col_dict_1.to_dict(orient="records")


def hold_same(df1, df2):
    compare = True
    for col in df1.columns:
        compare = (
            df1.loc[:, col].dropna() == df2.loc[:, col].dropna()
        ).all() and compare
        if not compare:
            break
    return compare


# --------------------------------------------------------------------------------
def roll_back(batch_first_events_avro_file, only_events=False):
    capture = Container(BLOB_CONTAINER_NAME)
    captureprocessed = Container(BLOB_CONTAINER_PROCESSED_NAME)

    captureprocessed.copy_blobs(
        capture,
        from_blob=batch_first_events_avro_file,
        delete=True,
    )

    capture.close()
    captureprocessed.close()

    if not only_events:
        intermediate_container = Container(INTERMEDIATE_CONTAINER_NAME)
        intermediateprocessed_container = Container(BACKUP_INTERMEDIATE_CONTAINER_NAME)

        intermediateprocessed_container.copy_blobs(intermediate_container)

        intermediate_container.close()
        intermediateprocessed_container.close()

#-------------------------------------------------------------------------------

def create_lock(message, container):
    fo = io.BytesIO()

    fo.write(json.dumps({"error_message": message}).encode())
    
    ContainerClient.upload_blob(
        container.container,
        name="lock",
        data=fo.getvalue(),
        overwrite=True
    )
    

def delete_lock(container):

    container.delete_blobs(pattern="^lock$")

# --------------------------------------------------------------------------------

def send_email(subject, msg): 
    msg = MIMEText(msg)
    msg['Subject'] = subject
    recipients = os.environ["recipients"].split(",")
    msg['To'] = ', '.join(recipients)#os.environ["recipients"]
    msg['From'] = os.environ["sender_email"]

    context = ssl.create_default_context()
    try:
        server = smtplib.SMTP(os.environ["smtp_server"], os.environ["port"])
        # server.set_debuglevel(True)
        server.starttls(context=context) 
        server.login(os.environ["user"], os.environ["password"])
        server.send_message(msg)
    except Exception as e:
        print(e)
    finally:
        server.quit() 

def send_htmlemail(subject, msghtml):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    recipients = os.environ["recipients"].split(",")
    msg['To'] = ', '.join(recipients)#os.environ["recipients"]
    msg['From'] = os.environ["sender_email"]
    partbody = MIMEText(msghtml,"html")
    msg.attach(partbody)

    # #Attach Document ##################
    # filename = "document.pdf"
    # with open(filename, "rb") as attachment:
    #     part = MIMEBase("application", "octet-stream")
    #     part.set_payload(attachment.read())
    # encoders.encode_base64(part)
    # part.add_header(
    #     "Content-Disposition",
    #     f"attachment; filename= {filename}",
    # )
    # msg.attach(part)
    # ######################################

    # #Attach Image ##################
    # fp = open('accesibilidad.png', 'rb') 
    # msgImage = MIMEImage(fp.read())
    # fp.close()
    # msgImage.add_header('Content-ID', '<image1>')
    # msg.attach(msgImage)
    ######################################

    context = ssl.create_default_context()
    try:
        server = smtplib.SMTP(os.environ["smtp_server"], os.environ["port"])
        # server.set_debuglevel(True)
        server.starttls(context=context)
        server.login(os.environ["user"], os.environ["password"])
        server.send_message(msg)
    except Exception as e:
        print(e)
    finally:
        server.quit() 


