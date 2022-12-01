GIT_VERSION ?= $(shell git describe --tags --always)

ARCH = $(shell uname -m)
# canonicalized names for target architecture
ifeq ($(ARCH),aarch64)
        override ARCH=arm64
endif
ifeq ($(ARCH),x86_64)
        override ARCH=amd64
endif

# supported deep learing frameworks
DLFRAMEWORKS = tf1 tf2 pytorch
# default framework
DLFRAMEWORK ?= tf2

DOCKERFILE ?= Dockerfile-$(DLFRAMEWORK)
GIT_VERSION ?= $(shell git describe --tags --always)
IMAGE ?= ai-benchmark-$(DLFRAMEWORK):$(GIT_VERSION)

# Check if the docker daemon is running in experimental mode (to get the --squash flag)
DOCKER_EXPERIMENTAL=$(shell docker version -f '{{ .Server.Experimental }}')
DOCKER_BUILD_ARGS?=
ifeq ($(DOCKER_EXPERIMENTAL),true)
DOCKER_BUILD_ARGS+=--squash
endif
ifneq ($(ARCH),amd64)
DOCKER_BUILD_ARGS+=--cpuset-cpus 0
endif

.PHONY:images
images:
	docker build $(DOCKER_BUILD_ARGS) --pull -t $(IMAGE) -f $(DOCKERFILE) .
