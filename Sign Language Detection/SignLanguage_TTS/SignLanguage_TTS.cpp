// SignLanguage_TTS.cpp : Defines the entry point for the application.
//

#include "SignLanguage_TTS.h"
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "sapi.lib")

void speakText(const std::wstring& text) {
	ISpVoice* pVoice = nullptr;

	if (FAILED(::CoInitialize(nullptr))) {
		std::cerr << "Failed to initialize COM." << std::endl;
		return;
	}

	if (SUCCEEDED(CoCreateInstance(CLSID_SpVoice, nullptr, CLSCTX_ALL, IID_ISpVoice, (void**)&pVoice))) {
		pVoice->Speak(text.c_str(), SPF_IS_XML, nullptr);
		pVoice->Release();
	}
	else {
		std::cerr << "Failed to create SAPI voice object." << std::endl;
	}

	// Cleanup COM
	::CoUninitialize();
}

int main()
{
	WSADATA wsaData;
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
		std::cerr << "WSAStartup failed!\n";
		return 1;
	}

	SOCKET client_socket = socket(AF_INET, SOCK_STREAM, 0);
	if (client_socket == INVALID_SOCKET) {
		std::cerr << "SOCKET CREATION FAILED!" << WSAGetLastError() << std::endl;

		WSACleanup();
		return -1;
	}

	sockaddr_in server_address;
	server_address.sin_family = AF_INET;
	server_address.sin_port = htons(PORT);
	inet_pton(AF_INET, SERVER_IP, &server_address.sin_addr);

	if (connect(client_socket, (sockaddr*)&server_address, sizeof(server_address)) == SOCKET_ERROR) {
		std::cerr << "Connection failed! Error: " << WSAGetLastError() << std::endl;
		closesocket(client_socket);
		WSACleanup();
		return 1;
	}
	
	char buffer[BUFFER_SIZE] = { 0 };
	int bytes_received;


	while (true) {
		memset(buffer, 0, BUFFER_SIZE);
		bytes_received = recv(client_socket, buffer, BUFFER_SIZE, 0);
		
		if (bytes_received > 0) {
			std::string received_text(buffer);
			std::cout << "Received: " << received_text << std::endl;

			std::wstring wText(buffer, buffer + bytes_received);

			speakText(wText);
		}
	}

	closesocket(client_socket);
	WSACleanup();

	return 0;
}

