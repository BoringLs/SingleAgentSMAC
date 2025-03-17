# Action

action len = 14

see in lib.const

# Obs（map “8m”）

type：list

len：n_agents

for each item in obs，it‘s array shaped like (80,)

80 = 4 direction whether can move

+8 enemy \* (can attack + dist + x_dist + y_dist + blood_percent)

+7 ally \* (can see + dist + x_dist + y_dist + blood_percent)

+own blood perccent

specially dist is relative dist , so true abs dist = dist \* 9(means sight_range)

# State（map “8m”）

map “8m” is 32 \* 32

max_distance_x = max_distance_y = 28

state.shape: (168,)

168 = 8 ally \* (blood_precent, weapon_cooldown, relative_x, relative_y）

+8 enemy \* (blood_precent, relative_x, relative_y）+

ally_last_action( 8 \* 14)

In state, true_x = (relative_x * 28) + 16

28 mean max_distance, 19 mean center oord
