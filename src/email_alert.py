
import os
import ssl
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("SENDER_APP_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")


def send_alert_email(event_type, tracking_id=None, image_path=None):
    if not all([SENDER_EMAIL, APP_PASSWORD, RECEIVER_EMAIL]):
        print("EMAIL ERROR: Email settings missing from .env")
        return False

    message = EmailMessage()
    message["Subject"] = f"SECURITY ALERT: {event_type}"
    message["From"] = SENDER_EMAIL.strip()
    message["To"] = RECEIVER_EMAIL.strip()

    timestamp = __import__("datetime").datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    message.set_content(
        f"Intelligent Video Surveillance System Alert\n\n"
        f"Event: {event_type}\n"
        f"Tracking ID: {tracking_id if tracking_id is not None else 'N/A'}\n"
        f"Time: {timestamp}\n\n"
        f"Please check the surveillance system for details."
    )

    # Attach the saved alert screenshot when available.
    if image_path and os.path.isfile(image_path):
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()

            message.add_attachment(
                image_data,
                maintype="image",
                subtype="jpeg",
                filename=os.path.basename(image_path)
            )
        except OSError as error:
            print("Could not attach screenshot:", error)

    try:
        context = ssl.create_default_context()

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(
                SENDER_EMAIL.strip(),
                APP_PASSWORD.replace(" ", "")
            )
            server.send_message(message)

        print(f"EMAIL SENT: {event_type}")
        return True

    except Exception as error:
        print(f"EMAIL FAILED for {event_type}: {type(error).__name__}: {error}")
        return False
