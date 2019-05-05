REGISTRY=$1
MODULE_PATH=$2

VERSION_FILE=$(cat "${MODULE_PATH}/VERSION")
IMAGE_NAME=${VERSION_FILE%:*}
IMAGE_TAG=${VERSION_FILE#*:}

DOCKER_IMAGE=${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}


MODES_ARR=()
for mode in main train inference deploy deploy_smart; do
	[[ -f "${MODULE_PATH}/src/${mode}.py" ]] && MODES_ARR+=( ${mode} )
done
MODES=${MODES_ARR[@]}

function get_file_content () {
	[[ -f "$1" ]] && echo $(base64 $1 | tr -d \\n) || echo ""
}

docker build \
	--label "VERSION=${DOCKER_IMAGE}" \
	--label "INFO=$(get_file_content "${MODULE_PATH}/plugin_info.json")" \
	--label "MODES=${MODES}" \
	--label "README=$(get_file_content "${MODULE_PATH}/README.md")" \
	--label "CONFIGS=$(get_file_content "${MODULE_PATH}/predefined_run_configs.json")" \
	--build-arg "MODULE_PATH=${MODULE_PATH}" \
	--build-arg "REGISTRY=${REGISTRY}" \
	--build-arg "TAG=${IMAGE_TAG}" \
	-f "${MODULE_PATH}/Dockerfile" \
	-t ${DOCKER_IMAGE} \
	.

echo "---------------------------------------------"
echo ${DOCKER_IMAGE}