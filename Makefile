setup-workspace: deps/sealir deps/spy


deps/sealir:
	bash scripts/checkout.sh https://github.com/sklam/sealir wip/updates_fixups deps/sealir

deps/spy:
	bash scripts/checkout.sh https://github.com/spylang/spy 1c04617d6916f875ddafffa9bd013a6508d28f79 deps/spy


build:
	pip install -e ./deps/sealir
	pip install -e './deps/spy[dev]'
	make -C ./deps/spy/spy/libspy
	pip install -e .


fmt:
	black -l79 ./nbcc


.PHONY: deps/spy deps/sealir
