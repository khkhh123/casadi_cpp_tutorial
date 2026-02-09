#include <casadi/casadi.hpp>
#include <iostream>

using namespace casadi;

int main() {
    // 1. 설정 및 파라미터 (동일)
    int N = 20;
    double dt = 0.1, m = 1500.0, v_ref = 100.0/3.6;

    // 2. Opti Stack 생성 (가장 큰 차이점!)
    Opti opti;

    // 3. 결정 변수 및 파라미터 정의 (인덱스 계산 필요 없음)
    auto X = opti.variable(N + 1); // 미래 속도 경로
    auto U = opti.variable(N);     // 미래 제어 입력
    auto v0 = opti.parameter();    // 현재 상태를 위한 파라미터

    // 4. 목적 함수 및 제약 조건 정의 (수학 기호 그대로 사용)
    MX J = 0;
    for (int k = 0; k < N; ++k) {
        // 비용 함수 누적
        J += pow(X(k) - v_ref, 2) + 1e-7 * pow(U(k), 2);

        // 동역학 제약 조건 (Euler integration)
        auto x_next = X(k) + (U(k) / m) * dt;
        opti.subject_to(X(k + 1) == x_next); 
    }
    opti.minimize(J);

    // 5. 경계 및 초기 제약 조건
    opti.subject_to(X(0) == v0);
    opti.subject_to(opti.bounded(-3000.0, U, 3000.0)); // 입력 범위 제한 한 줄로 끝

    // 6. 솔버 설정
    Dict opts;
    opts["qpsol"] = "qrqp";
    opts["print_header"] = false;
    opti.solver("sqpmethod", opts);

    // 7. 시뮬레이션 루프
    double current_v = 0.0;
    std::cout << "Time | Velocity | Control Input" << std::endl;

    for (int i = 0; i < 200; ++i) {
        // 현재 속도 주입 및 이전 해를 초기 추측값으로 사용
        opti.set_value(v0, current_v);
        
        // 솔루션 도출
        auto sol = opti.solve();

        // 결과 추출 (인덱스 번호가 아닌 변수 이름으로 접근!)
        double control_u = double(sol.value(U(0)));

        current_v += (control_u / m) * dt;
        printf("%3.1f s | %6.2f km/h | %7.1f N\n", i * dt, current_v * 3.6, control_u);
    }

    return 0;
}