# Generated by Django 4.2.1 on 2023-05-25 08:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chimedAi_web', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='users',
            name='is_staff',
            field=models.BooleanField(default=False),
        ),
    ]
