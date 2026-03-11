"""
Как встроить в Telegram-бота.

from moderation_engine import HybridModerator
moderator = HybridModerator()

text = update.effective_message.text or update.effective_message.caption or ""
verdict = moderator.moderate_text(text)
if verdict.flagged:
    # удалить сообщение / замьютить / логировать
    ...

# Для фото:
verdict = moderator.moderate_image(temp_file_path, caption=text)
"""
