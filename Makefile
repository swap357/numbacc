setup-workspace: deps/sealir deps/spy


deps/sealir:
	bash scripts/checkout.sh https://github.com/sklam/sealir wip/updates_fixups deps/sealir

deps/spy:
	bash scripts/checkout.sh https://github.com/spylang/spy 49304dec85d7c1f872c562b2c7a7f2ae893f800f deps/spy


build:
	pip install -e ./deps/sealir
	pip install -e './deps/spy[dev]'
	make -C ./deps/spy/spy/libspy
	pip install -e .


fmt:
	black -l79 ./nbcc


.PHONY: deps/spy deps/sealir
