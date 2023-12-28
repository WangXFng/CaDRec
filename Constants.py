PAD = 0

DATASET = "douban-book"  # Yelp2018  Gowalla  Foursquare ml-1M douban-book
ABLATIONs = {'w/oWeSch', 'w/oPopDe', 'w/oSA', 'w/oNorm', 'w/oUSpec', 'w/oHgcn', 'w/oDisen'}
ABLATION = 'Full'

user_dict = {
    'ml-1M': 6038,
    'douban-book': 12859,
    'Gowalla': 18737,
    'Yelp2018': 31668,
    'Foursquare': 7642
}

item_dict = {
    'ml-1M': 3533,
    'douban-book': 22294,
    'Gowalla': 32510,
    'Yelp2018': 38048,
    'Foursquare': 28483
}
ITEM_NUMBER = item_dict.get(DATASET)
USER_NUMBER = user_dict.get(DATASET)


print('Dataset:', DATASET, '#User:', USER_NUMBER, '#ITEM', ITEM_NUMBER)
print('ABLATION: ', ABLATION)


beta_dict = {
    'ml-1M': 0.42,  # best 0.42,  # 0.5,  # 0.8  # 3629
    'douban-book': 0.07,  # best 0.07, 0.05,
    'Gowalla': 0.1,
    'Yelp2018': 0.25,  # best 0.25,
    'Foursquare': 0.5,  # best 0.5
}
BETA_1 = beta_dict[DATASET]
if ABLATION == 'w/oPopDe' or ABLATION == 'w/oDisen': BETA_1 = 0

# During ablation study, in below cases, the popularity features will be over weighted, leading to over low accuracies.
if ABLATION == 'w/oSA' and DATASET == 'Yelp2018': BETA_1 = 0.01
if ABLATION == 'w/oSA' and DATASET == 'Yelp': BETA_1 = 0
if ABLATION == 'w/oHgcn' and DATASET == 'Gowalla': BETA_1 = 0
if ABLATION == 'OlyHGCN' and DATASET == 'Yelp2018': BETA_1 = 0.05

print('BETA_1', BETA_1)






