# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app
COPY ./gradio_demo.py /app/gradio_demo.py
COPY ./teaching_staff.py /app/teaching_staff.py
COPY ./requirements.txt /app/requirements.txt
COPY ./api_key.txt /app/api_key.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt

EXPOSE 80

# Run app.py when the container launches
CMD ["python", "gradio_demo.py"]