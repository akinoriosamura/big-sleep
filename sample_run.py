from big_sleep import Imagine


target = 'target.txt'
with open(target) as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]

import pdb;pdb.set_trace()

for text in lines:
    model = Imagine(
        text = text,
        lr = 5e-2,
        save_every = 10,
        save_progress = True
    )

    epochs = 2
    iters = 10
    for epoch in range(epochs):
        print('epoch: ', epoch)
        for i in range(iters):
            model.train_step(epoch, i)

            if i == 0 or i % model.save_every != 0:
                continue