# ROS Noetic Docker Environment

This repository contains a Docker setup for ROS Noetic development.

## Prerequisites

- Docker
- Docker Compose

## Building and Running

1. Build the Docker image:
```bash
docker-compose build
```

2. Start the container:
```bash
docker-compose up -d
```

3. Enter the container:
```bash
docker-compose exec ros_noetic bash
```

## Usage

Once inside the container, you can:
- Use ROS commands as normal
- Build your workspace with `catkin build`
- Run ROS nodes and launch files

## Stopping the Container

To stop the container:
```bash
docker-compose down
```

## Notes

- The workspace is mounted as a volume, so changes made inside the container will persist on your host machine
- The container has network access to your host machine
- GUI applications are supported through X11 forwarding 