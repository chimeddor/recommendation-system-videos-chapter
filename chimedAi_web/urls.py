from django.urls import path, include
from . import views
from django.contrib.auth import views as auth_views
from django.conf.urls.static import static
from django.conf import settings
#from django.views.generic import TemplateView

urlpatterns = [
    path('admin/clearcache/', include('clearcache.urls')),
    path("", views.base, name="base"),
    path('loginpage/', views.login_page, name='loginpage'),
    path('loginview/', views.login_view, name='loginview'),
    
    #register 페이지로 이동
    path('register/',views.register_view, name='register'),
    
    path('signuppage/', views.sign_up_page, name='signuppage'),
    
    path('signup/', views.sign_up, name='signup'),

    path('signupnext/',views.signup_next, name='signupnext'),
    
    path('signupnextpage/',views.signup_next_page, name='signupnextpage'),

    path('newvideopage/', views.New_video_page, name='newvideopage'),
    path('insertvideo/', views.Insert_video, name='insertvideo'),
    path('Searched_video/', views.Searched_video, name='Searched_video'),
    # path('watch/<path:video_l>', views.watch, name='watch'),
    path('watch/<str:youtube_id>/', views.watch, name='watch'),
    # path('chapter_watch/<str:youtube_id>/', views.chapter_watch, name='watch'),
    # path('my_video_list/', views.my_video_list, name="my_video_list"),
    path('', views.professor, name="professor"),
    path('prof_profile/', views.Prof_profile, name="prof_profile"),
    path('prof_channels/',views.Prof_channels, name="prof_channels"),
    
    # 자기 정보 수정 요청
    path('informationChange/<str:id>/', views.Prof_change, name="informationChange"),
    
    path('error/', views.error, name = "error"),
    path('search_recommandation/', views.Search_recommendation, name="search_recommandation"),
    path('logout/', views.LogoutView, name='logout'),
    
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)




    # path('login/', auth_views.LoginView.as_view(template_name='login/login.html'), name='login'),




