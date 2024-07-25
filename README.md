“chapter level lecture video recommendation system using Graph Neural Network"

license - mozilla public license (MPL)
django platform에 그래프 신경망과 그 외에 다른 알고리즘들을 활용함.

<p><b>--- 환경 설치 ---</b></p>
python 3.8.10
django 4.2.1
mysql  Ver 15.1 Distrib 10.6.15-MariaDB, for debian-linux-gnu (x86_64) using readline 5.2

<p><b>--- 변경된 사항 있는 경우 다음 명령어를 터미너에서 실행 ---</b></p>

<p>- python3 manage.py makemigrations</p>
<p> - python3 manage.py migrate</p>

<p><b>--- 시스템 사용 방법.---</b></p>

<p> - 데이터 베이스를 장고 administration 페이지에서 관리 가능.</p>
<p> - 장고 administration 페이지에서 사용자 및 강의 동영상 관리 가능.</p>
<p>- 검색 부분을 통해 질문하면 챕터 추천하며, 챕터를 클릭 시 해당되는 강의 동영상으로 이동.</p>


<p><b>--- 시스템을 실행하는 명령어 ---</b></p>

<p>- python3 manage.py runserver 0: *</p>

<p><b>--- 시스템 ---</b></p>

0. youtube link
1. 자동으로 강의 동영상에 대한 챕터를 생성
2. TF-IDF
3. k-means++ Clustering
4. GNN
5. Cosine similarity

1)   

![Screenshot 2024-03-25 at 21 08 12](https://github.com/chimeddor/recommendation-system-videos-chapter/assets/53028417/e8ae8793-ad38-478b-b068-17414e526d0d)

2)

![Screenshot 2024-03-25 at 21 09 05](https://github.com/chimeddor/recommendation-system-videos-chapter/assets/53028417/0c090f1d-aec8-4257-8d98-78ec79fabbaa)

3)

![Screenshot 2024-03-25 at 21 09 46](https://github.com/chimeddor/recommendation-system-videos-chapter/assets/53028417/cbaf189f-8572-4a3a-9d5f-3cc437c20f73)
