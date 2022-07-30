FROM tensorflow/tensorflow:2.9.1-gpu-jupyter
ENV HOME=/project
WORKDIR /project

COPY requirements.txt . 
RUN pip install \
    --no-cache-dir \
    -r requirements.txt \
    && rm -rf requirements.txt

COPY src/ .
COPY entrypoint.sh .

ENTRYPOINT [ "/project/entrypoint.sh" ]
