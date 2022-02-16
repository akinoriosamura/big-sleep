from big_sleep import Imagine


target = 'target.txt'
with open(target) as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
lines = lines * 5
# import pdb;pdb.set_trace()

for text_id, text in enumerate(lines):
    print('start : ', text)
    model = Imagine(
        text = text,
        lr = 5e-2,
        save_every = 19,
        save_progress = True
    ).cuda()

    epochs = 3
    iters = 20
    for epoch in range(epochs):
        print('epoch: ', epoch)
        for i in range(iters):
            model.train_step(epoch, i, text_id)

            if i == 0 or i % model.save_every != 0:
                continue
    print('finish : ', text)
