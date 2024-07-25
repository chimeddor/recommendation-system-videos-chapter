from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model
from django.db.models import Q  # Add this line to import the Q object

class EmailOrUsernameBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        UserModel = get_user_model()
        try:
            user = UserModel.objects.get(email=username)
            if user.check_password(password):
                return user
            else:
                return None
        except UserModel.DoesNotExist:
            return None
        

    def get_user(self, user_id):
        try:
            UserModel = get_user_model()  # 기존의 UserModel을 가져옴
            user = UserModel.objects.get(id=user_id)
            return user
        except UserModel.DoesNotExist:
            return None




# class EmailOrUsernameBackend(ModelBackend):
#     def authenticate(self, request, email_or_username=None, password=None, **kwargs):
#         UserModel = get_user_model()
#         try:
#             user = UserModel.objects.get(Q(email=email_or_username) | Q(username=email_or_username))
#         except UserModel.DoesNotExist:
#             return None
#         else:
#             if user.check_password(password):
#                 return user
#         return None

#     def get_user(self, user_id):
#         UserModel = get_user_model()
#         try:
#             return UserModel.objects.get(pk=user_id)
#         except UserModel.DoesNotExist:
#             return None
