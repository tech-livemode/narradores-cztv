import smtplib
from email.message import EmailMessage
from config import settings

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def Send_Mail(assunto: str, mensagem: str):
    # monta a mensagem
    msg = EmailMessage()
    msg["From"] = settings.username
    msg["To"] = ", ".join(settings.destinatarios)
    msg["Subject"] = assunto
    msg.set_content(mensagem)

    # conexão segura com auto-fechamento
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(settings.username, settings.password)
            smtp.send_message(msg)
        print("E-mail enviado com sucesso!")
    except Exception as e:
        print(f"Erro ao enviar e-mail: {e}")


if __name__ == "__main__":
    Send_Mail("Teste", "Encontramos o narrador!")