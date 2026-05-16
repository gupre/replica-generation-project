"""
chat.py
Интерактивный чат-интерфейс для модели техподдержки.
Запуск: python chat.py --checkpoint ./checkpoints/checkpoint-best
"""

import argparse
import sys
import os
import logging

logging.basicConfig(level=logging.WARNING)  # тихо при чате

# Цвета для терминала
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
GRAY   = "\033[90m"


def print_banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════╗
║          Ubuntu Support Bot  (GPT-2 fine-tuned)      ║
║  Команды: /clear  /history  /config  /help  /quit    ║
╚══════════════════════════════════════════════════════╝{RESET}
""")


def print_help():
    print(f"""
{YELLOW}Доступные команды:{RESET}
  {BOLD}/clear{RESET}           — очистить историю диалога
  {BOLD}/history{RESET}         — показать текущую историю
  {BOLD}/config{RESET}          — показать/изменить параметры генерации
  {BOLD}/temp <float>{RESET}    — изменить temperature (0.1–1.5)
  {BOLD}/topp <float>{RESET}    — изменить top_p (0.5–1.0)
  {BOLD}/topk <int>{RESET}      — изменить top_k (1–200)
  {BOLD}/candidates <n>{RESET}  — показать N вариантов ответа
  {BOLD}/help{RESET}            — эта справка
  {BOLD}/quit{RESET} или Ctrl+C — выход
""")


def format_turn(role: str, text: str) -> str:
    if role == "user":
        return f"{GREEN}{BOLD}You:{RESET} {text}"
    else:
        return f"{CYAN}{BOLD}Bot:{RESET} {text}"


def run_chat(bot, num_candidates: int = 1):
    history = []
    print_banner()
    print(f"{GRAY}Введите вопрос по Ubuntu/Linux. Для справки введите /help{RESET}\n")

    while True:
        try:
            user_input = input(f"{GREEN}{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{YELLOW}Завершение работы...{RESET}")
            break

        if not user_input:
            continue

        # ── Команды ──
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd in ("/quit", "/exit", "/q"):
                print(f"{YELLOW}До свидания!{RESET}")
                break

            elif cmd == "/clear":
                history.clear()
                print(f"{GRAY}История очищена.{RESET}\n")

            elif cmd == "/history":
                if not history:
                    print(f"{GRAY}История пуста.{RESET}\n")
                else:
                    print(f"\n{YELLOW}История диалога:{RESET}")
                    for i, turn in enumerate(history):
                        role = "You" if i % 2 == 0 else "Bot"
                        print(f"  [{i+1}] {role}: {turn}")
                    print()

            elif cmd == "/config":
                print(f"""
{YELLOW}Текущие параметры генерации:{RESET}
  temperature      = {bot.temperature}
  top_p            = {bot.top_p}
  top_k            = {bot.top_k}
  max_new_tokens   = {bot.max_new_tokens}
  repetition_pen.  = {bot.repetition_penalty}
  context_window   = {bot.context_window}
  candidates       = {num_candidates}
""")

            elif cmd == "/temp" and len(parts) == 2:
                try:
                    val = float(parts[1])
                    bot.update_config(temperature=max(0.1, min(2.0, val)))
                    print(f"{GRAY}temperature = {bot.temperature}{RESET}\n")
                except ValueError:
                    print(f"{RED}Неверное значение.{RESET}\n")

            elif cmd == "/topp" and len(parts) == 2:
                try:
                    val = float(parts[1])
                    bot.update_config(top_p=max(0.1, min(1.0, val)))
                    print(f"{GRAY}top_p = {bot.top_p}{RESET}\n")
                except ValueError:
                    print(f"{RED}Неверное значение.{RESET}\n")

            elif cmd == "/topk" and len(parts) == 2:
                try:
                    val = int(parts[1])
                    bot.update_config(top_k=max(1, min(500, val)))
                    print(f"{GRAY}top_k = {bot.top_k}{RESET}\n")
                except ValueError:
                    print(f"{RED}Неверное значение.{RESET}\n")

            elif cmd == "/candidates" and len(parts) == 2:
                try:
                    num_candidates = max(1, min(5, int(parts[1])))
                    print(f"{GRAY}Будет показано {num_candidates} вариант(а).{RESET}\n")
                except ValueError:
                    print(f"{RED}Неверное значение.{RESET}\n")

            elif cmd == "/help":
                print_help()

            else:
                print(f"{RED}Неизвестная команда: {cmd}. Введите /help.{RESET}\n")

            continue

        # ── Генерация ответа ──
        history.append(user_input)

        print(f"{GRAY}Генерация...{RESET}", end="\r")

        try:
            responses = bot.generate(history, num_candidates=num_candidates)
        except Exception as e:
            print(f"{RED}Ошибка генерации: {e}{RESET}\n")
            history.pop()
            continue

        if num_candidates == 1:
            response = responses[0]
            print(f"\r{format_turn('bot', response)}\n")
            history.append(response)
        else:
            print(f"\r{CYAN}{BOLD}Bot (варианты):{RESET}")
            for i, r in enumerate(responses, 1):
                print(f"  [{i}] {r}")
            # Выбор пользователем
            try:
                choice = input(f"{GRAY}Выберите [1-{len(responses)}] или Enter для первого: {RESET}").strip()
                idx = (int(choice) - 1) if choice.isdigit() else 0
                idx = max(0, min(idx, len(responses) - 1))
            except (ValueError, KeyboardInterrupt):
                idx = 0
            response = responses[idx]
            print(f"{GRAY}Выбрано: {response}{RESET}")
            history.append(response)
            print()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Чат с Ubuntu Support Bot")
    parser.add_argument(
        "--checkpoint", type=str, default="./checkpoints/checkpoint-best",
        help="Путь к директории с обученной моделью"
    )
    parser.add_argument("--device", type=str, default=None,
                        help="cuda | cpu | mps (авто по умолчанию)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--context_window", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=1,
                        help="Сколько вариантов ответа предлагать (1-5)")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"{RED}Checkpoint не найден: {args.checkpoint}{RESET}")
        print("Сначала запустите train.py для обучения модели.")
        sys.exit(1)

    from inference import SupportBot

    bot = SupportBot(
        checkpoint_dir=args.checkpoint,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        context_window=args.context_window,
    )

    run_chat(bot, num_candidates=args.candidates)
