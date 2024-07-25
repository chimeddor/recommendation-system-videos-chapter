from typing import Any
from django.db import models
from datetime import date
from phonenumber_field.modelfields import PhoneNumberField
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AbstractUser,BaseUserManager,UserManager


class CustomUserManager(BaseUserManager):
    def _create_user(self, email, password, **extra_fields):
        extra_fields.setdefault('data_exists', False)
        if not email:
            raise ValueError("The Email field must be set")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user
    def create_user(self, email=None, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', False)
        extra_fields.setdefault('is_superuser', False)
        return self._create_user(email, password, **extra_fields)
    
    def create_superuser(self, email=None, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self._create_user(email, password, **extra_fields)

class Users(AbstractUser):
    student = 0
    professor = 1
    choice_role = (
        (student, 'student'),
        (professor, 'professor'),
    )
    # USERNAME_FIELD = ['username']
    # REQUIRED_FIELDS = ('email', )

    username = models.CharField(max_length=200, unique=True)
    phone_number = PhoneNumberField()
    email = models.EmailField(unique=True)
    role = models.PositiveSmallIntegerField(choices=choice_role, blank=False, null=True)
    password = models.CharField(max_length=200)
    signed_date = models.DateField(default=date.today)
    data_exists = models.BooleanField(default=False)
    is_staff=models.BooleanField(default=False)
    # objects = UserManager()
    objects = CustomUserManager()
 
class Video(models.Model):
    title = models.CharField(max_length=100)
    description = models.TextField()
    youtube_id  = models.CharField(max_length=200, unique=True)
    # video_l = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(
        get_user_model(),
        on_delete=models.CASCADE, #해당 강사 삭제 시 따라 영상이 삭제
        related_name='uploaded_videos',
        limit_choices_to={'role': Users.professor},
    )
    def save(self, *args, **kwargs):
        self.uploaded_by = Users.objects.get(role=Users.professor)
        super().save(*args, **kwargs)

# Create your models here.
# INSERT INTO chimedAi_web_video (title, description, video_link) VALUES ("What Does A Machine Learning Engineer At Amazon Do?", "","opb2Bq4Qvyo&t");
