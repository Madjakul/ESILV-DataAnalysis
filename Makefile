MODULE := o_classifier

BLUE='\033[0;34m'
NC='\033[0m' # No Color

install-requirements:
	pip3 install -r requirements.txt

run:
	@python3 -m $(MODULE)

test:
	@pytest

lint:
	@echo "\n${BLUE}Running Pylint against source and test files...${NC}\n"
	@pylint --rcfile=setup.cfg **/*.py
	@echo "\n${BLUE}Running Bandit against source files...${NC}\n"
	@bandit -r --ini setup.cfg

build:
	@docker build -t ${MODULE}:latest .
	@echo "\n${BLUE}Running the app...${NC}\n"
	@docker run -t ${MODULE} ls -l

clean:
	rm -rf .pytest_cache .coverage .pytest_cache coverage.xml

.PHONY: clean test

docker-clean:
	@docker system prune -f --filter "label=name=$(MODULE)"
