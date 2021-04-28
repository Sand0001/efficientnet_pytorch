class_dict = {'bloody_car': 1, 'politics_drug': 2, 'politics_military_emblem': 3, 'politics_party_emblem': 4,
 'politics_weapon_tank': 5, 'bloody_carcrash': 6, 'politics_flag_uniform': 7, 'politics_military_flag': 8,
 'politics_party_flag': 9, 'zc_normal_cartoon': 10, 'bloody_cartoon': 11, 'politics_gamble': 12,
 'politics_military_flag_part': 13, 'politics_uniform': 14, 'zc_normal_other': 15, 'bloody_limb': 16,
 'politics_gamble_game': 17, 'politics_national_emblem': 18, 'politics_weapon_airplane': 19,
 'zc_normal_renxiang': 20, 'bloody_other': 21, 'politics_league_emblem': 22, 'politics_national_flag': 23,
 'politics_weapon_birdfarm': 24, 'zc_normal_text': 25, 'politics_boom_fire': 26, 'politics_league_flag': 27,
 'politics_national_flag_part': 28, 'politics_weapon_gun': 29, 'zc_normal_uniform_cartoon': 30, 'politics_coin': 31,
 'politics_map': 32, 'politics_paper_money': 33, 'politics_weapon_knife': 34,'politics_100_years':35,'normal_four_wheel_vehicle':36,
              'politics_tricycle':37}
num2name={1: 'bloody_car', 2: 'politics_drug', 3: 'politics_military_emblem', 4: 'politics_party_emblem', 5: 'politics_weapon_tank', 6: 'bloody_carcrash',
          7: 'politics_flag_uniform', 8: 'politics_military_flag', 9: 'politics_party_flag', 10: 'zc_normal_cartoon', 11: 'bloody_cartoon',
          12: 'politics_gamble', 13: 'politics_military_flag_part', 14: 'politics_uniform', 15: 'zc_normal_other', 16: 'bloody_limb', 17: 'politics_gamble_game', 18: 'politics_national_emblem', 19: 'politics_weapon_airplane', 20: 'zc_normal_renxiang', 21: 'bloody_other', 22: 'politics_league_emblem',
          23: 'politics_national_flag', 24: 'politics_weapon_birdfarm', 25: 'zc_normal_text', 26: 'politics_boom_fire', 27: 'politics_league_flag',
          28: 'politics_national_flag_part', 29: 'politics_weapon_gun', 30: 'zc_normal_uniform_cartoon', 31: 'politics_coin',
          32: 'politics_map', 33: 'politics_paper_money', 34: 'politics_weapon_knife',35:'politics_100_years',36:'normal_four_wheel_vehicle',37:'politics_tricycle'}

class_dict2 ={'bloody_car': 0, 'bloody_carcrash': 1, 'bloody_cartoon': 2, 'bloody_limb': 3, 'bloody_other': 4, 'politics_boom_fire': 5, 'politics_coin': 6,
              'politics_drug': 7, 'politics_flag_uniform': 8, 'politics_gamble': 9, 'politics_gamble_game': 10, 'politics_league_emblem': 11,
              'politics_league_flag': 12, 'politics_map': 13, 'politics_military_emblem': 14, 'politics_military_flag': 15, 'politics_military_flag_part': 16, 'politics_national_emblem': 17, 'politics_national_flag': 18,
              'politics_national_flag_part': 19, 'politics_paper_money': 20, 'politics_party_emblem': 21, 'politics_party_flag': 22, 'politics_uniform': 23, 'politics_weapon_airplane': 24, 'politics_weapon_birdfarm': 25, 'politics_weapon_gun': 26,
              'politics_weapon_knife': 27, 'politics_weapon_tank': 28, 'zc_normal_cartoon': 29, 'zc_normal_other': 30,
              'zc_normal_renxiang': 31,'zc_normal_text':32,
'zc_normal_uniform_cartoon':33,'politics_100_years':34,'normal_four_wheel_vehicle':35,
              'politics_tricycle':36}


root_dir = '/data/fengjing/terr_test/baokong_train'
checkpoint_dir = '/data/fengjing/checkpoint_b3_add_100years'
batchsize = 48
size= 224
class_num = 38
num_workers = 16
optimizer = 'SGD'
# data_list = ['bloody_car', 'politics_drug', 'politics_military_emblem', 'politics_party_emblem', 'politics_weapon_tank',
#              'bloody_carcrash', 'politics_flag_uniform', 'politics_military_flag', 'politics_party_flag', 'zc_normal_cartoon',
#              'bloody_cartoon', 'politics_gamble', 'politics_military_flag_part', 'politics_uniform', 'zc_normal_other', 'bloody_limb',
#              'politics_gamble_game', 'politics_national_emblem', 'politics_weapon_airplane', 'zc_normal_renxiang', 'bloody_other',
#              'politics_league_emblem', 'politics_national_flag', 'politics_weapon_birdfarm', 'zc_normal_text', 'politics_boom_fire',
#              'politics_league_flag', 'politics_national_flag_part', 'politics_weapon_gun',
#              'zc_normal_uniform_cartoon', 'politics_coin', 'politics_map', 'politics_paper_money', 'politics_weapon_knife','politics_100_years',
#              'normal_four_wheel_vehicle','politics_tricycle']

data_list = ['bloody_car' ,'politics_drug',]
epochs = 100
gpus = '0,1'
seed = 2