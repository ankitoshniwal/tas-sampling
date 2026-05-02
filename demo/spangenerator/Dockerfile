FROM golang:1.23-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o spangenerator .

FROM alpine:3.19
WORKDIR /app
COPY --from=builder /app/spangenerator .
ENTRYPOINT ["./spangenerator"]
