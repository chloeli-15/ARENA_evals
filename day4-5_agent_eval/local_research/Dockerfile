# A dockerfile must always start by importing the base image.
# We use the keyword 'FROM' to do that.
# In our example, we want import the python image.
# So we write 'python' for the image name and 'latest' for the version.
FROM python:3.9

# Set the working directory within the container. The path depends on the base image you choose.
WORKDIR /root

# Copy task-relevant code into the container using 'COPY'. We should copy any task files and model files necessary for running your LLM agent into the "root" folder. These will be made INACCESSIBLE TO THE AGENT. The first parameter 'main.py' is the name of the file on the host. The second parameter './' is the path where to put the file on the image, in this case the folder "/root" specified by WORKDIR.
# COPY main.py ./ 
# COPY agent.py ./
# COPY requirements.txt ./
COPY assets/ ./ 

# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables if your LM requires any
# ENV MY_ENV_VAR=my_value

# Create the agent user with a home directory at /home/agent
# RUN useradd -m -s /bin/bash -u 1000 agent


# Run application code when the container launches
CMD [ "python", "./test.py" ]


# Build the Docker image in terminal
# $ docker build -t my-python-app . #The ’-t’ option allows you to define the name of your image. In our case we have chosen ’python-test’ but you can put what you want.
# $ docker run -it --rm --name my-running-app my-python-app #put the name of your image after ‘docker run’.
