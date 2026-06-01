"""Terminal tic-tac-toe against the trained policy-value MCTS agent."""

from agent.policy_value_agent import PolicyValueAgent
from game.board import Board
from play.logic import agent_move, load_agent, outcome_message, symbol


def render(board: Board) -> None:
    labels = {0: ".", 1: "X", 2: "O"}
    for row in range(3):
        cells = [labels[board.cells[row * 3 + col]] for col in range(3)]
        print(" ".join(cells))


def prompt_move(board: Board) -> int:
    while True:
        raw = input("Your move (1-9): ").strip()
        if not raw.isdigit():
            print("Enter a number from 1 to 9.")
            continue
        action = int(raw) - 1
        if action in board.legal_moves():
            return action
        print("That cell is not available.")


def play_game(human_player: int, agent: PolicyValueAgent) -> None:
    board = Board()
    current = 1

    while not board.is_terminal():
        render(board)
        if current == human_player:
            action = prompt_move(board)
        else:
            print("Agent is thinking...")
            action = agent_move(agent, board, current)

        board = board.apply_move(action, current)
        current = 2 if current == 1 else 1

    render(board)
    print(outcome_message(board, human_player))


def main() -> None:
    agent = load_agent()
    print("Play tic-tac-toe against the trained policy-value MCTS agent.")
    while True:
        choice = input("Play as X or O? [X/o/q]: ").strip().lower()
        if choice in {"q", "quit"}:
            break
        human_player = 2 if choice == "o" else 1
        print(f"You are {symbol(human_player)}.")
        play_game(human_player, agent)
        again = input("Play again? [Y/n]: ").strip().lower()
        if again in {"n", "no"}:
            break


if __name__ == "__main__":
    main()
