import argparse

parser = argparse.ArgumentParser(
    prog='Relationship transformation',
    description='Project for ICCV 2023 xxx',
)
parser.add_argument('--project_name', default='Relationship transformation', type=str)
parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'])

'''
args for train
'''


'''
args for test
'''


'''
args for model
'''

'''
args for metrics
'''
args = parser.parse_args()