sudo apt install curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install bazel
sudo apt update && sudo apt full-upgrade
sudo apt install bazel-1.0.0


curl https://github.com/bazelbuild/bazel/releases/download/5.0.0-pre.20210831.2/bazel-5.0.0-pre.20210831.2-installer-linux-x86_64.sh
bazel-1.0.0-installer-linux-x86_64.sh
chmod +x bazel-1.0.0-installer-linux-x86_64.sh
./bazel-1.0.0-installer-linux-x86_64.sh --user