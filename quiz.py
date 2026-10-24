questions = [
    ("Capital of Pakistan?", "islamabad"),
    ("Capital of France?", "paris"),
    ("2 + 2?", "4"),
]


def ask(question, correct):
    answer = input(question + " ")
    if answer.strip().lower() == correct:
        print("Correct")
        return 1
    print("Wrong")
    return 0



# Main game loop
score = 0
for question, correct in questions:
    score += ask(question, correct)

print("\nYou scored", score, "out of", len(questions))
