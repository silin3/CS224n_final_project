from evaluation import test_sonnet
score = test_sonnet(
    test_path='predictions/sonnets-gpt2-10-1e-05_v1.txt',
    gold_path='data/TRUE_sonnets_held_out_dev.txt'
)
print('CHRF =', score)