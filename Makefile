setup-workspace: deps/sealir deps/spy


deps/sealir:
	bash scripts/checkout.sh https://github.com/sklam/sealir wip/updates_fixups deps/sealir

deps/spy:
	bash scripts/checkout.sh https://github.com/spylang/spy 40a15774d1b8ef28f988ba02f1707121852afa6c deps/spy


build:
	pip install -e ./deps/sealir
	pip install -e './deps/spy[dev]'
	make -C ./deps/spy/spy/libspy
	pip install -e .


fmt:
	black -l79 ./nbcc


.PHONY: deps/spy deps/sealir
