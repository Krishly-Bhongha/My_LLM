from trainer import train_one_merge

NUM_MERGES = 10000

for i in range(NUM_MERGES):

    success = train_one_merge()

    if not success:
        print(f"Stopped after {i} merges.")
        break