import uuid
import random


class User(object):

    def __init__(self, user_id, username, password, token=None):
        self.user_id = user_id
        self.username = username.lower()
        self.password = password
        self.token = token


class UserStore(object):

    def __init__(self):
        self.users_by_username = {}

    def register_user(self, username, password):
        if username == '' or password == '':
            raise Exception("Username or password not specified")

        if username in self.users_by_username.keys():
            raise Exception("Username already exists")

        if username in self.users_by_username.keys():
            raise Exception("Username already exists")

        user_id = str(uuid.uuid1())
        user = User(user_id, username, password)
        self.users_by_username[user.username] = user
        return user

    def signin(self, username, password):
        if username not in self.users_by_username:
            raise Exception("Wrong username or password")
        user = self.users_by_username[username]
        if user.password != password:
            raise Exception("Wrong username or password")
        # generate very unsafe token
        user.token = ''.join(random.sample(range(0, 9), 10))
        return user

    def signout(self, username, token):
        if username not in self.users_by_username:
            raise Exception("Wrong username or password")
        user = self.users_by_username[username]
        if user.token != token:
            raise Exception("Wrong username or password")
        # generate very unsafe token
        user.token = None
        return user

    def verify_token(self, username, token):
        if username not in self.users_by_username:
            raise Exception("User does not exist")
        user = self.users_by_username[username]
        if user.token is None or user.token != token:
            raise Exception("Token is invalid")
        return True
