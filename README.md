<h1><b>Chapter level lecture video recommendation system using Graph Neural Network</b></h1>
<h4>논문[paper]-KJC(Korea Journal Central) <a href="https://journal.kci.go.kr/jksci/archive/articleView?artiId=ART003104517">Improving Accuracy of Chapter-level Lecture Video Recommendation System using Keyword Cluster-based Graph Neural Networks</a></4>

<h4>논문[paper]-RISS <a href="https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=8880060ffec5650bffe0bdc3ef48d419&outLink=K">Graph Neural Network를 활용한 챕터 수준의 강의 동영상 추천 시스템 = Chapter-level lecture video recommendation system using Graph Neural Network</a></4>

<h4>논문[paper]-scienceON <a href="https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=DIKO0016835391&fbclid=IwZXh0bgNhZW0CMTEAAR0N098y0pUjimMzz6kju2Mw_rh9xQrOLJjI0Cb3tYyWNDxf6dmER_YOvic_aem_BiH8cKASkCIdlzfZ82PACg#;">Graph Neural Network를 활용한 챕터 수준의 강의 동영상 추천 시스템 = Chapter-level lecture video ,recommendation system using Graph Neural Network</a></4>
<p></p>
<br>
<p>license - mozilla public license (MPL)</p><br>
<p>django platform에 그래프 신경망과 그 외에 다른 알고리즘들을 활용</p>

<h3> 1. 환경 설치 </h3>
    <p>1.1 python 3.8.10</p>
    <p>1.2 django 4.2.1</p>
    <p>1.3 mysql  Ver 15.1 Distrib 10.6.15-MariaDB, for debian-linux-gnu (x86_64) using readline 5.2</p>

<h3><b> 2. 가상 환경 필요한 경우 </b></h3>

    2.1 python3 -m venv my-venv

<h3><b>3. 라이브러리 설치하는 명령어 </b></h3>

    3.1 pip install django-libraryname

<h3><b>4. 변경된 사항 있는 경우 다음 명령어를 터미너에서 실행 </b></h3>

    4.1 python3 manage.py makemigrations
    4.2 python3 manage.py migrate

<h3><b>5. 시스템 사용 방법 </b></h3>
    <p> 5.1 데이터 베이스를 장고 administration 페이지에서 관리 가능</p>
    <p> 5.2 장고 administration 페이지에서 사용자 및 강의 동영상 관리 가능</p>
    <p> 5.3 검색 부분을 통해 질문하면 챕터 추천하며, 챕터를 클릭 시 해당되는 강의 동영상으로 이동</p>


<h3><b>6. 시스템을 실행하는 명령어 </b></h3>
    
    6.1 python3 manage.py runserver 0:port

<h3><b>7. 시스템 구성 </b></h3>
    <p> 7.1 Speech-To-Text </p>
    <p> 7.2 챕터 생성</p>
    <p> 7.3 TF-IDF(챕터간 유사도 측정)</p>
    <p> 7.4 k-means++ Clustering(유사한 챕터들끼리 클러스터링)</p>
    <p> 7.5 GNN(Graph Neural Network) </p>
    <p> 7.6 Cosine similarity </p>

1)   

![Screenshot 2024-03-25 at 21 08 12](https://github.com/chimeddor/recommendation-system-videos-chapter/assets/53028417/e8ae8793-ad38-478b-b068-17414e526d0d)

2)

![Screenshot 2024-03-25 at 21 09 05](https://github.com/chimeddor/recommendation-system-videos-chapter/assets/53028417/0c090f1d-aec8-4257-8d98-78ec79fabbaa)

3)

![Screenshot 2024-03-25 at 21 09 46](https://github.com/chimeddor/recommendation-system-videos-chapter/assets/53028417/cbaf189f-8572-4a3a-9d5f-3cc437c20f73)
