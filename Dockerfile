FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install git+https://github.com/facebookresearch/segment-anything.git
RUN pip install --no-cache-dir \
    torch \
    numpy \
    opencv-python \
    Flask \
    Werkzeug
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

CMD ["python", "main.py"]

