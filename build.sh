CURRENT_PATH="$(cd `dirname $0`; pwd)"
VERSION=$(cat $CURRENT_PATH/version)

./prebuild.sh
echo "version of deep is ${VERSION}"

echo "start build docker image:"
docker buildx build --platform  linux/amd64 -t deep:${VERSION} .