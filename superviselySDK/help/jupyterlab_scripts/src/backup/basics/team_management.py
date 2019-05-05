# coding: utf-8
import supervisely_lib as sly
import os

address = os.environ['SERVER_ADDRESS']
token = os.environ['API_TOKEN']

print("Server address: ", address)
print("Your API token: ", token)

api = sly.Api(address, token)

# get all user teams
user_teams = api.team.get_list()
print("Founded {} teams:".format(len(user_teams)))
for team in user_teams:
    print("{:<5}{:<15s}".format(team.id, team.name))
# access to TeamInfo fields
team = user_teams[0]
print("Team information:")
print(team)

# create new team
team_name = 'test_team'
team_description = 'test description'
if api.team.exists(team_name):
    team_name = api.team.get_free_name(team_name)
created_team = api.team.create(team_name, team_description)
print('Team (id={}, name={!r}) has been successfully created: '.format(created_team.id, created_team.name))
print(created_team)


# get team info by name
team_name = 'test_team'
team = api.team.get_info_by_name(team_name)
if team is None:
    print("Team {!r} not found".format(team_name))
else:
    print(team)


# get team info by id
user_teams = api.team.get_list()
some_team_id = user_teams[0].id
team = api.team.get_info_by_id(some_team_id)
if team is None:
    print("Team with id={!r} not found".format(some_team_id))
else:
    print(team)


# update team name, description, or both
new_name = 'my_super_team'
new_description = 'super description'
updated_team = api.team.update(created_team.id, new_name, new_description)
print("Before update: {}".format(created_team))
print("After  update: {}".format(updated_team))
