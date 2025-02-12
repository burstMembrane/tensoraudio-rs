.PHONY: test test-verbose  setup-env source-env

OS := $(shell uname -s)

env:
	. bin/${OS}-env.sh
test:
	. bin/${OS}-env.sh && cargo test
test-verbose:
	. bin/${OS}-env.sh && cargo test -- --nocapture

build:
	. bin/${OS}-env.sh && cargo build

setup-env:
	. bin/${OS}-env.sh

source-env:
	. bin/${OS}-env.sh 
