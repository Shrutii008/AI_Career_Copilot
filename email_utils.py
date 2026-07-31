"""
email_utils.py
-----------------
Sends the resume analysis outputs (PDF report, cover letter, tailored
resume PDF) via email, using SMTP with an App Password.

Reads two existing environment variables (from .env):
    EMAIL_ADDRESS       - the sender's email address
    EMAIL_APP_PASSWORD  - an App Password for that account (not the
                           regular account password)

Defaults to Gmail's SMTP server, since "App Password" is Gmail/Google
terminology. Override with SMTP_SERVER / SMTP_PORT in .env if you're
using a different provider (e.g. Outlook, Yahoo).
"""

import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape as _html_escape

from dotenv import load_dotenv

load_dotenv()

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

# Brand palette, kept in sync with the app's CSS variables and the PDF
# report's colors. Emails render in dozens of different mail clients
# with very inconsistent dark-mode support, so — like the PDF — this
# uses a light card on a neutral background rather than a literal
# dark-mode layout; the indigo/purple/cyan brand accents are what tie
# it visually to the app and the PDF report.
_BRAND_NAME = "AI Career Copilot"
_PRIMARY = "#4F46E5"
_SECONDARY = "#7C3AED"
_HIGHLIGHT = "#06B6D4"
_TEXT_DARK = "#111827"
_TEXT_GREY = "#6B7280"
_BG = "#F3F1FA"
_CARD_BG = "#FFFFFF"
_BORDER = "#E5E7EB"


def _build_html_body(body_text: str) -> str:
    """
    Wrap a plain-text email body in a simple branded HTML template
    (light card, brand-gradient header, muted footer). Falls back
    gracefully — the plain-text version is always sent alongside this
    as the 'alternative' MIME part, so nothing is lost for clients
    that don't render HTML.
    """
    # Preserve line breaks and blank-line paragraph spacing; escape
    # user/AI-generated content to avoid breaking the markup.
    safe = _html_escape(body_text or "").replace("\r\n", "\n")
    paragraphs = [p.strip() for p in safe.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [safe]
    body_html = "".join(
        f'<p style="margin:0 0 14px 0; white-space:pre-line;">{p}</p>'
        for p in paragraphs
    )

    return f"""\
<html>
  <body style="margin:0; padding:0; background:{_BG}; font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{_BG}; padding:32px 16px;">
      <tr>
        <td align="center">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:560px; background:{_CARD_BG}; border:1px solid {_BORDER}; border-radius:16px; overflow:hidden;">
            <tr>
              <td style="background:linear-gradient(135deg,{_PRIMARY},{_SECONDARY}); padding:22px 28px;">
                <span style="color:#ffffff; font-size:13px; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; opacity:0.9;">🧭 {_BRAND_NAME}</span>
              </td>
            </tr>
            <tr>
              <td style="padding:28px 28px 8px 28px; color:{_TEXT_DARK}; font-size:14.5px; line-height:1.6;">
                {body_html}
              </td>
            </tr>
            <tr>
              <td style="padding:18px 28px 26px 28px;">
                <div style="height:3px; width:48px; background:{_HIGHLIGHT}; border-radius:2px; margin-bottom:14px;"></div>
                <span style="color:{_TEXT_GREY}; font-size:12px;">Sent by {_BRAND_NAME} — your AI career companion.</span>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>"""


def email_credentials_available() -> bool:
    """True if both required env vars are set."""
    return bool(os.getenv("EMAIL_ADDRESS")) and bool(os.getenv("EMAIL_APP_PASSWORD"))


def email_status_message() -> str:
    """Human-readable status, for a sidebar/banner or inline warning."""
    if email_credentials_available():
        return f"✅ Email sending configured ({os.getenv('EMAIL_ADDRESS')})"
    missing = []
    if not os.getenv("EMAIL_ADDRESS"):
        missing.append("EMAIL_ADDRESS")
    if not os.getenv("EMAIL_APP_PASSWORD"):
        missing.append("EMAIL_APP_PASSWORD")
    return f"⚠️ Email not configured — add {', '.join(missing)} to your .env file"


def send_email_report(
    recipient_email: str,
    subject: str,
    body_text: str,
    attachment_bytes: bytes,
    attachment_filename: str,
    attachment_mime_subtype: str = "pdf",
):
    """
    Send an email with a single file attachment via SMTP.

    Returns (success: bool, message: str) — never raises, so callers
    can show the message directly via st.success()/st.error().
    """
    sender = os.getenv("EMAIL_ADDRESS")
    app_password = os.getenv("EMAIL_APP_PASSWORD")

    if not sender or not app_password:
        return False, (
            "Email is not configured. Add EMAIL_ADDRESS and "
            "EMAIL_APP_PASSWORD to your .env file."
        )

    if not recipient_email or "@" not in recipient_email or "." not in recipient_email.split("@")[-1]:
        return False, "Please enter a valid recipient email address."

    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient_email
        msg["Subject"] = subject

        # multipart/alternative: plain-text body (unchanged, always
        # present) + a branded HTML version. Mail clients that support
        # HTML show the branded version; everything else falls back to
        # the exact same plain text as before.
        body_alt = MIMEMultipart("alternative")
        body_alt.attach(MIMEText(body_text, "plain"))
        try:
            body_alt.attach(MIMEText(_build_html_body(body_text), "html"))
        except Exception:
            pass  # HTML templating is cosmetic only; never block sending on it
        msg.attach(body_alt)

        part = MIMEApplication(attachment_bytes, _subtype=attachment_mime_subtype)
        part.add_header(
            "Content-Disposition", "attachment", filename=attachment_filename
        )
        msg.attach(part)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(sender, app_password)
            server.sendmail(sender, recipient_email, msg.as_string())

        return True, f"✅ Email sent successfully to {recipient_email}!"

    except smtplib.SMTPAuthenticationError:
        return False, (
            "⚠️ Email authentication failed. Check EMAIL_ADDRESS and "
            "EMAIL_APP_PASSWORD — this must be an App Password, not your "
            "regular account password."
        )
    except smtplib.SMTPRecipientsRefused:
        return False, f"⚠️ The recipient address '{recipient_email}' was refused by the mail server."
    except smtplib.SMTPConnectError:
        return False, f"⚠️ Could not connect to {SMTP_SERVER}:{SMTP_PORT}. Check your network/SMTP settings."
    except (smtplib.SMTPException, OSError) as exc:
        return False, f"⚠️ Failed to send email: {exc}"