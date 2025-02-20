import subprocess

def test_player():
    # Start the player script
    process = subprocess.Popen(
        ["python", "lasker_morris_player.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    # Send player ID
    process.stdin.write("blue\n")
    process.stdin.flush()

    # Simulate opponent moves
    opponent_moves = ["h1 d1 r0", "d1 d2 e3"]
    for move in opponent_moves:
        process.stdin.write(move + "\n")
        process.stdin.flush()

        # Read player's response
        response = process.stdout.readline().strip()
        print(f"Player responded: {response}")

    # Close the process
    process.stdin.close()
    process.stdout.close()
    process.terminate()

if __name__ == "__main__":
    test_player()
