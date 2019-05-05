# How to create Superviely plugin

This tutorial walks you through how to create Supervisely Plugin. It will show you how to add the necessary files and structure to create the plugin, how to build the plugin, how to upload it to remote registry and how to add it to your account in Supervisely.

# Plugin file structure

Create the following file structure:

```
.
├── src
│   └── main.py
├── supervisely_lib (optional)
│   └── ...
├── Dockerfile
├── LICENSE
├── plugin_info.json
├── predefined_run_configs.json (optional)
├── README.md
└── VERSION
```

## Sources
Folder `src` contains your source codes. If you use `supervisely_lib`, put its directory to the root of the plugin directory.


## Dockerfile

Paragraph from [Docker documentation](https://docs.docker.com/engine/reference/builder/):

Docker can build images automatically by reading the instructions from a `Dockerfile`. A `Dockerfile` is a text document that contains all the commands a user could call on the command line to assemble an image. Using `docker build` users can create an automated build that executes several command-line instructions in succession. 

At the end of your `Dockerfile` you have to put `COPY` command to copy source codes to the directory `/workdir` in docker image. Also you may need to update `PYTHONPATH` environment variable.

```
COPY . /workdir
ENV PYTHONPATH /workdir:/workdir/src:/workdir/supervisely_lib/worker_proto:$PYTHONPATH
```  

## License

It’s important for every plugin to include a license. This tells users who install your package the terms under which they can use your package. For help picking a license, see https://choosealicense.com/. Once you have chosen a license, open LICENSE and enter the license text. 


## Plugin info

`plugin_info.json` defines the name and description fo your plugin. File contains JSON object with the following structure:

```json
{
	"title": "<Plugin name>",
	"description": "<Plugin description>",
	"type": "<Plugin type>"
}
```

Supervisely supports several types of plugins: 
* `import` - all plugins with this type will be available in "Import" page and will be used to import data to the system. Import means convert data from some format to supervisely format 
* `dtl` - plugins to manipulate data with Data Transformation Language scripts
* `custom` - custom plugin that can do allmost anything you need 
* `architecture` - plugin defines how to train/inference/deploy neural network


## Readme

README file will be available on Plugin Page. Example:

![](https://i.imgur.com/YjNwmiP.png)


## Version

Version file contains the name of the docker image (without docker registry name) and its tag in the following format `<docker image>:<tag>`. Example:

```
nn-icnet:4.0.0
``` 

## Predefined run configs

If your plugin takes a specific configuration as input, it is better to prepare some default examples in JSON file `predefined_run_configs.json` in the following structure:

```json
[
  {
    "title": "<template title>",
    "type": "<type of the plugin or mode(train/inference) in case of neural networks>",
    "config": {"custom configuration": "is here"}
  },
  {
    "title": "<template title>",
    "type": "<type of the plugin or mode(train/inference) in case of neural networks>",
    "config": {"custom configuration": "is here"}
  }
]
```


# How to build plugin

To build plugin docker image execute the following shell script `build_plugin.sh` in the plugin root directory.

```sh
./build_plugin.sh <my docker registry> .
```


`build_plugin.sh` script content: 

```sh
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
```  


This script runs `docker build` command and attaches technical files (readme, version, predefined configs) as labels in docker image. Then Supervisely platform will automatically extract all labels during importing plugin.


# How to add plugin to Supervisely

Go to "Plugnis" page and press "Add" button:

![](https://i.imgur.com/uvBF7y2.png) 

Enter plugin title and docker image and press "Create" button:

![](https://i.imgur.com/DJsuyJ4.png) 

As a result new plugin will appear in your "Plugins" page:

![](https://i.imgur.com/YjNwmiP.png)