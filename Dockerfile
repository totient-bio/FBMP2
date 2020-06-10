FROM python:3.6.5

RUN mkdir /fbmp2
RUN mkdir /fbmp2/fbmp2
COPY fbmp2/* /fbmp2/fbmp2/
COPY setup.py /fbmp2/
COPY requirements.txt /fbmp2/
COPY README.md /fbmp2/

RUN pip install --upgrade pip
RUN pip install /fbmp2
